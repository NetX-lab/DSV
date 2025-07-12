#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <tuple>
#include <numeric>
#include <string>
#include <cassert>
#include <thread>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hardware_config.h"

namespace py = pybind11;

// Enum for deployment strategy
enum class DeploymentStrategy {
    HCP_First_Intra_Node,
    SCP_First_Intra_Node
};

// Helper function to convert DeploymentStrategy to string
std::string deployment_strategy_to_string(DeploymentStrategy strategy) {
    switch (strategy) {
        case DeploymentStrategy::HCP_First_Intra_Node:
            return "HCP-First, Intra-Node";
        case DeploymentStrategy::SCP_First_Intra_Node:
            return "SCP-First, Intra-Node";
        default:
            return "Unknown";
    }
}

template<typename T>
std::string vec2d_to_string(const std::vector<std::vector<T>>& vec) {
    std::string s = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        s += "[";
        for (size_t j = 0; j < vec[i].size(); ++j) {
            s += std::to_string(vec[i][j]);
            if (j + 1 < vec[i].size()) s += ", ";
        }
        s += "]";
        if (i + 1 < vec.size()) s += ", ";
    }
    s += "]";
    return s;
}

template<typename T>
std::string vec_to_string(const std::vector<T>& vec) {
    std::string s = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        s += std::to_string(vec[i]);
        if (i + 1 < vec.size()) s += ", ";
    }
    s += "]";   
    return s;
}

template <typename T>
T estimate_flash_attention_flops(int seq_len, int head_dim, int g_h, int g_s, const std::vector<T>& kv_sparsity) {
    T total_flops = 0.0;
    int local_head_num = static_cast<int>(kv_sparsity.size());

    for (int i = 0; i < local_head_num; ++i) {
        T sparsity = kv_sparsity[i];
        T head_flops = (1.0 - sparsity) * seq_len/g_s * seq_len * head_dim * 2.0 * 2.0; // first 2 for QK and AV; second 2 for flops count;
        T softmax_flops = (1.0 - sparsity) * seq_len/g_s * seq_len * 4.0; // 
        total_flops += head_flops + softmax_flops;
    }
    return total_flops;
}

// Return both the sparsity group and the index list of the sparsity group for each RANK in a HCP group
template <typename T>
std::tuple<std::vector<std::vector<T>>, std::vector<std::vector<int>>> split_sparsity_groups(const std::vector<T>& sparsity, int HCP_GROUP_SIZE) {
    std::vector<std::vector<T>> sparsity_value_list_per_group(HCP_GROUP_SIZE);
    std::vector<std::vector<int>> sparsity_index_list_per_group(HCP_GROUP_SIZE);

    std::vector<T> comp_burden_sum_per_group(HCP_GROUP_SIZE, 0); 

    std::vector<std::pair<int, T>> indexed_comp_burden;
    for (size_t i = 0; i < sparsity.size(); ++i) {
        indexed_comp_burden.emplace_back(i, 1.0 - sparsity[i]);
    }

    std::sort(indexed_comp_burden.begin(), indexed_comp_burden.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    for (const auto& [idx, value] : indexed_comp_burden) {
        int min_group = 0;
        for (int i = 1; i < HCP_GROUP_SIZE; ++i) {
            if (comp_burden_sum_per_group[i] < comp_burden_sum_per_group[min_group]) {
                min_group = i;
            }
        }
        sparsity_value_list_per_group[min_group].push_back(1.0 - value);
        sparsity_index_list_per_group[min_group].push_back(idx);
        comp_burden_sum_per_group[min_group] += value;
    }
    return std::make_tuple(sparsity_value_list_per_group, sparsity_index_list_per_group);
}

// Template structure for optimal parallel configuration
template <typename T>
struct OptimalParallelConfig {
    int HCP_GROUP_SIZE; // The GPU Number for a HCP group
    int SCP_GROUP_SIZE; // The GPU Number for a SCP group
    std::vector< std::vector<int> > Reallocated_Sparsity_Idx_Per_Group; // The Index list of the sparsity group for each RANK in a HCP group
    T total_cost; // Total cost (compute time + communication time)
    T compute_time; // Compute time
    T communication_time; // Communication time
    DeploymentStrategy deployment_strategy; // Deployment strategy
};

// Template class for Hybrid Sparsity Optimizer
template <typename T>
class HybridSparsityOptimizer {
private:
    ServerConfig server_config; // server hardware configuration
    int HEAD_NUM; // Number of attention heads
    int SEQ_LEN; // Sequence length
    int HEAD_DIM; // Attention head dimension
    T MEM_BUFFER_PER_GPU; // Per-GPU memory buffer limit in B for the attention module 
    std::string DATA_TYPE; // Data type for the attention module
    int bytes_per_element; // Bytes per element for the attention module
    T compute_efficiency; // Compute efficiency factor (0.0 to 1.0)
    T comm_efficiency; // Communication efficiency factor (0.0 to 1.0)
    T scp_overhead; // SCP communication overhead in seconds (for index computation, etc.)

    std::mutex best_config_mtx; // Mutex for thread-safe updates to the best configuration

    // Compute the compute time for each GPU
    T compute_time(const std::vector<T>& sparsity, int g_h, int g_s) {
        T total_flops = estimate_flash_attention_flops(SEQ_LEN, HEAD_DIM, g_h, g_s, sparsity);
        T gpu_compute_flops = server_config.get_compute_flops_per_second(DATA_TYPE) * compute_efficiency;
        return total_flops / gpu_compute_flops;
    }

    // Compute communication cost (support intra-machine and inter-machine deployments)
    T compute_communication_cost(int g_h, int g_s, const std::vector<T>& sparsity, const std::vector<std::vector<T>>& sparsity_value_list_per_group, bool hcp_first_intra_node, bool scp_first_intra_node) {
            // - The HCP group involves 4 All-to-All communications.
            // - The SCP group involves 2 Allgather communications.
            // - If a group spans multiple nodes, the hierarchical communication cost must be considered.
            // - For each group, first check whether its GPUs are distributed across different nodes.
            //     - If a group spans multiple nodes and there are at least 2 ranks for the group within a node,
            //       hierarchical communication is required.
            //     - Otherwise, Situation 1:intra-node bandwidth can be used directly, as communication is purely within a single node. 
            //     - Situation 2: inter-node bandwidth can be used directly, as communication is purely across different nodes.  


        assert (hcp_first_intra_node != scp_first_intra_node);

        int hcp_gpu_num_within_node, hcp_gpu_num_across_nodes;
        int scp_gpu_num_within_node, scp_gpu_num_across_nodes;

        if (hcp_first_intra_node) {
            hcp_gpu_num_within_node = std::min(g_h, server_config.gpus_per_node);
            hcp_gpu_num_across_nodes =  g_h / hcp_gpu_num_within_node; 
            scp_gpu_num_within_node = std::min(server_config.gpus_per_node/hcp_gpu_num_within_node, 1 ); 
            scp_gpu_num_across_nodes = g_s / scp_gpu_num_within_node;
        } else if (scp_first_intra_node) { 
            scp_gpu_num_within_node = std::min(g_s, server_config.gpus_per_node);
            scp_gpu_num_across_nodes = g_s / scp_gpu_num_within_node;
            hcp_gpu_num_within_node = std::min(server_config.gpus_per_node/scp_gpu_num_within_node, 1);
            hcp_gpu_num_across_nodes = g_h / hcp_gpu_num_within_node;
        }

        int data_volume_per_gpu = SEQ_LEN * HEAD_DIM * HEAD_NUM / server_config.total_gpus; 

        T hcp_comm_time = 0.0, scp_comm_time = 0.0;
        T bandwidth_intra_node = server_config.get_intra_node_bandwidth_in_bytes_per_second() * comm_efficiency;
        T bandwidth_inter_node = server_config.get_inter_node_bandwidth_in_bytes_per_second() * comm_efficiency;

        if (hcp_gpu_num_across_nodes > 1) {
            T inter_node_data_volume = data_volume_per_gpu/g_h * (hcp_gpu_num_within_node * (hcp_gpu_num_across_nodes - 1)); 
            T intra_node_data_volume = data_volume_per_gpu/g_h * (hcp_gpu_num_within_node - 1);
            T inter_node_comm = inter_node_data_volume * 4.0 * bytes_per_element / bandwidth_inter_node; // 4 for 2 All-to-All communications
            T intra_node_comm = intra_node_data_volume * 4.0 * bytes_per_element / bandwidth_intra_node;
            T hierarchical_comm = std::max(inter_node_comm, intra_node_comm);
            hcp_comm_time = hierarchical_comm;
        } else {
            T intra_node_data_volume = data_volume_per_gpu/g_h * (hcp_gpu_num_within_node - 1);
            T intra_node_comm = intra_node_data_volume * 4.0 * bytes_per_element / bandwidth_intra_node;
            hcp_comm_time = intra_node_comm;
        }

        T avg_sparsity_sum_across_hcp_group = 0.0;

        for (int i = 0; i < g_h; ++i) {
            avg_sparsity_sum_across_hcp_group += std::accumulate(sparsity_value_list_per_group[i].begin(), sparsity_value_list_per_group[i].end(), static_cast<T>(0.0));
        }

        avg_sparsity_sum_across_hcp_group /= HEAD_NUM;

        if (scp_gpu_num_across_nodes > 1) {
            T inter_node_data_volume = data_volume_per_gpu * (1-avg_sparsity_sum_across_hcp_group) * (scp_gpu_num_within_node * (scp_gpu_num_across_nodes - 1));
            T intra_node_data_volume = data_volume_per_gpu * (1-avg_sparsity_sum_across_hcp_group) * (scp_gpu_num_within_node - 1);
            T inter_node_comm = inter_node_data_volume * 2.0 * bytes_per_element / bandwidth_inter_node; // 2 for 2 Allgather communications
            T intra_node_comm = intra_node_data_volume * 2.0 * bytes_per_element / bandwidth_intra_node;
            T hierarchical_comm = std::max(inter_node_comm, intra_node_comm);
            scp_comm_time = hierarchical_comm;
        } else {
            T intra_node_data_volume = data_volume_per_gpu * (1-avg_sparsity_sum_across_hcp_group) * (scp_gpu_num_within_node - 1);
            T intra_node_comm = intra_node_data_volume * 2.0 * bytes_per_element / bandwidth_intra_node;
            scp_comm_time = intra_node_comm;
        } 

        // Add SCP overhead for sequence parallel communication when g_s > 1
        // This accounts for index computation, sparse KV gathering setup, etc.
        if (g_s > 1) {
            scp_comm_time += scp_overhead;
        }

        return hcp_comm_time + scp_comm_time;
    }

    // Check if memory usage is within the limit
    bool check_memory(int HCP_GROUP_SIZE, int SCP_GROUP_SIZE, const std::vector<T>& sparsity) {
        T avg_sparsity = std::accumulate(sparsity.begin(), sparsity.end(), static_cast<T>(0.0)) / sparsity.size();
        T hcp_memory = 4.0 * bytes_per_element * SEQ_LEN * HEAD_DIM * HEAD_NUM / HCP_GROUP_SIZE;
        T scp_memory = 2.0 * bytes_per_element * SEQ_LEN * HEAD_DIM * HEAD_NUM * (1.0 - avg_sparsity) * (SCP_GROUP_SIZE - 1);
        return (hcp_memory + scp_memory <= MEM_BUFFER_PER_GPU);
    }

    // Evaluate configurations within a specific range of g_h
    void evaluate_range(int start_g_h, int end_g_h, const std::vector<T>& sparsity, OptimalParallelConfig<T>& local_best_config, bool verbose = false) {
        for (int g_h = start_g_h; g_h <= end_g_h; ++g_h) {
            int g_s = server_config.total_gpus / g_h;

            if (g_h != 1 && g_s * g_h != server_config.total_gpus) continue; // HCP degree must be 1 or even

            if (!check_memory(g_h, g_s, sparsity)) {
                if (verbose) {
                    std::cout << "\n" << std::string(60, '=') << "\n";
                    std::cout << "\033[1;41m"  
                            << "MEMORY LIMIT EXCEEDED"
                            << "\033[0m" << "\n";
                    std::cout << std::string(60, '=') << "\n";
                    std::cout << "\033[1;31m"  
                            << "Configuration rejected:\n"
                            << "  • HCP Group Size (g_h): " << g_h << "\n"
                            << "  • SCP Group Size (g_s): " << g_s << "\n"
                            << "  • Total GPU Number: " << server_config.total_gpus << "\n"
                            << "  • Reason: Memory requirements exceed per-GPU buffer limit"
                            << "\033[0m" << "\n";
                    std::cout << std::string(60, '=') << "\n" << std::endl;
                }
                continue;
            }

            if (verbose) {
                std::cout << "\033[1;34m" << ">>> Evaluating Configuration: g_h = " << g_h << ", g_s = " << g_s << "\033[0m" << std::endl;
            }

            T max_compute_time = 0.0;

            std::vector<std::vector<T>> sparsity_value_list_per_group;
            std::vector<std::vector<int>> sparsity_index_list_per_group;
            std::tie(sparsity_value_list_per_group, sparsity_index_list_per_group) = split_sparsity_groups(sparsity, g_h);

            if (verbose) {
                std::cout << "\033[1;35m" << "Debug Information:" << "\033[0m\n";
                std::cout << "  • Sparsity Value List: " << vec2d_to_string<double>(sparsity_value_list_per_group) << "\n";
                std::cout << "  • Sparsity Index List: " << vec2d_to_string<int>(sparsity_index_list_per_group) << "\n" << std::endl;
            }

            for (int i = 0; i < g_h; ++i) {
                max_compute_time = std::max(max_compute_time, compute_time(sparsity_value_list_per_group[i], g_h, g_s));
            }

            // Evaluate both deployment strategies
            T comm_cost_hcp_first_intra_node = compute_communication_cost(g_h, g_s, sparsity, sparsity_value_list_per_group, true, false);
            T comm_cost_scp_first_intra_node = compute_communication_cost(g_h, g_s, sparsity, sparsity_value_list_per_group, false, true);

            // Choose the better strategy and calculate total cost accordingly
            OptimalParallelConfig<T> candidate_config;
            if (comm_cost_hcp_first_intra_node < comm_cost_scp_first_intra_node) {
                T total_cost = max_compute_time + comm_cost_hcp_first_intra_node;
                candidate_config = {g_h, g_s, sparsity_index_list_per_group, total_cost, max_compute_time, comm_cost_hcp_first_intra_node, DeploymentStrategy::HCP_First_Intra_Node};
            } else {
                T total_cost = max_compute_time + comm_cost_scp_first_intra_node;
                candidate_config = {g_h, g_s, sparsity_index_list_per_group, total_cost, max_compute_time, comm_cost_scp_first_intra_node, DeploymentStrategy::SCP_First_Intra_Node};
            }

            if (verbose) {
                std::cout << "\n" << std::string(50, '-') << "\n";
                std::cout << "\033[1;32m" << "CANDIDATE CONFIGURATION" << "\033[0m\n";
                std::cout << std::string(50, '-') << "\n";
                std::cout << "\033[1;36m" << "Parallelization Strategy:" << "\033[0m\n";
                std::cout << "  • HCP Group Size: " << candidate_config.HCP_GROUP_SIZE << "\n";
                std::cout << "  • SCP Group Size: " << candidate_config.SCP_GROUP_SIZE << "\n";
                std::cout << "\033[1;33m" << "Performance Metrics:" << "\033[0m\n";
                std::cout << "  • Total Cost: " << candidate_config.total_cost << " seconds\n";
                std::cout << "  • Compute Time: " << candidate_config.compute_time << " seconds\n";
                std::cout << "  • Communication Time: " << candidate_config.communication_time << " seconds\n";
                std::cout << "\033[1;36m" << "Configuration Details:" << "\033[0m\n";
                std::cout << "  • Deployment Strategy: " << deployment_strategy_to_string(candidate_config.deployment_strategy) << "\n";
                std::cout << "  • Sparsity Index Groups: " << vec2d_to_string<int>(candidate_config.Reallocated_Sparsity_Idx_Per_Group) << "\n";
                std::cout << "\033[1;37m" << "Efficiency Factors:" << "\033[0m\n";
                std::cout << "  • Compute Efficiency: " << compute_efficiency << " (" << (compute_efficiency * 100) << "%)\n";
                std::cout << "  • Communication Efficiency: " << comm_efficiency << " (" << (comm_efficiency * 100) << "%)\n";
                std::cout << "  • SCP Overhead: " << scp_overhead << " seconds\n";
                std::cout << std::string(50, '-') << "\n" << std::endl;
            }

            // Update local best configuration
            if (candidate_config.total_cost < local_best_config.total_cost) {
                local_best_config = candidate_config;
            }
        }
    }

public:
    // Constructor - using ServerConfig
    HybridSparsityOptimizer(
        const ServerConfig& config,
        int HEAD_NUM, int SEQ_LEN, int HEAD_DIM, T MEM_BUFFER_PER_GPU,
        std::string DATA_TYPE, T compute_efficiency = 1.0, T comm_efficiency = 1.0, T scp_overhead = 0.0)
        : server_config(config),
          HEAD_NUM(HEAD_NUM),
          SEQ_LEN(SEQ_LEN),
          HEAD_DIM(HEAD_DIM),
          MEM_BUFFER_PER_GPU(MEM_BUFFER_PER_GPU),
          DATA_TYPE(DATA_TYPE),
          bytes_per_element(0),
          compute_efficiency(compute_efficiency),
          comm_efficiency(comm_efficiency),
          scp_overhead(scp_overhead)
    {
        if (DATA_TYPE == "fp16" || DATA_TYPE == "bf16") {
            bytes_per_element = 2;
        } else if (DATA_TYPE == "fp8") {
            bytes_per_element = 1;
        } else {
            throw std::invalid_argument("Invalid data type: " + DATA_TYPE + 
                                        "\nSupported data types: fp16, bf16, fp8");
        }
        
        // Validate efficiency factors
        if (compute_efficiency <= 0.0 || compute_efficiency > 1.0) {
            throw std::invalid_argument("Compute efficiency must be in range (0.0, 1.0], got: " + std::to_string(compute_efficiency));
        }
        if (comm_efficiency <= 0.0 || comm_efficiency > 1.0) {
            throw std::invalid_argument("Communication efficiency must be in range (0.0, 1.0], got: " + std::to_string(comm_efficiency));
        }
        if (scp_overhead < 0.0) {
            throw std::invalid_argument("SCP overhead must be non-negative, got: " + std::to_string(scp_overhead));
        }
    }


    // Run the optimization using a specified number of threads and sparsity
    OptimalParallelConfig<T> optimize(const std::vector<T>& sparsity, int num_threads, bool verbose = false) {
        std::vector<std::thread> threads;
        std::vector<OptimalParallelConfig<T>> local_best_configs(num_threads, {0, 0, std::vector<std::vector<int>>(), std::numeric_limits<T>::max(), 0.0, 0.0, DeploymentStrategy::HCP_First_Intra_Node});
        int total_g_h = server_config.total_gpus; // Total range of g_h to evaluate
        int chunk_size = (total_g_h + num_threads - 1) / num_threads; // Divide the range into chunks

        // Launch threads to evaluate different ranges of g_h
        for (int i = 0; i < num_threads; ++i) {
            int start_g_h = i * chunk_size + 1;
            int end_g_h = std::min((i + 1) * chunk_size, total_g_h);
            if (start_g_h <= end_g_h) {
                threads.emplace_back(&HybridSparsityOptimizer::evaluate_range, this, start_g_h, end_g_h, std::cref(sparsity), std::ref(local_best_configs[i]), verbose);
            }
        }

        // Join all threads
        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }

        // Find the global best configuration across all threads
        OptimalParallelConfig<T> global_best_config = {0, 0, std::vector<std::vector<int>>(), std::numeric_limits<T>::max(), 0.0, 0.0, DeploymentStrategy::HCP_First_Intra_Node};
        for (const auto& config : local_best_configs) {
            if (config.total_cost < global_best_config.total_cost) {
                global_best_config = config;
            }
        }

        return global_best_config;
    }

    // get server configuration information
    ServerConfig get_server_config() const {
        return server_config;
    }

    // get compute efficiency
    T get_compute_efficiency() const {
        return compute_efficiency;
    }

    // get communication efficiency
    T get_comm_efficiency() const {
        return comm_efficiency;
    }

    // get SCP overhead
    T get_scp_overhead() const {
        return scp_overhead;
    }

    // print server information
    void print_server_info() const {
        HardwareConfigManager::print_server_info(server_config);
    }
};

// Python binding
PYBIND11_MODULE(hybrid_optimizer, m) {
    py::enum_<DeploymentStrategy>(m, "DeploymentStrategy")
        .value("HCP_First_Intra_Node", DeploymentStrategy::HCP_First_Intra_Node)
        .value("SCP_First_Intra_Node", DeploymentStrategy::SCP_First_Intra_Node)
        .export_values();

    py::class_<OptimalParallelConfig<double>>(m, "OptimalParallelConfig")
        .def_readonly("HCP_GROUP_SIZE", &OptimalParallelConfig<double>::HCP_GROUP_SIZE)
        .def_readonly("SCP_GROUP_SIZE", &OptimalParallelConfig<double>::SCP_GROUP_SIZE)
        .def_readonly("Reallocated_Sparsity_Idx_Per_Group", &OptimalParallelConfig<double>::Reallocated_Sparsity_Idx_Per_Group)
        .def_readonly("total_cost", &OptimalParallelConfig<double>::total_cost)
        .def_readonly("compute_time", &OptimalParallelConfig<double>::compute_time)
        .def_readonly("communication_time", &OptimalParallelConfig<double>::communication_time)
        .def_readonly("deployment_strategy", &OptimalParallelConfig<double>::deployment_strategy)
        .def("__str__", [](const OptimalParallelConfig<double>& config) {
            return std::string(50, '=') + "\n" +
                   "\033[1;42m\033[1;37m OPTIMAL PARALLEL CONFIGURATION \033[0m\n" +
                   std::string(50, '=') + "\n" +
                   "\033[1;36mParallelization Strategy:\033[0m\n" +
                   "  • HCP Group Size: " + std::to_string(config.HCP_GROUP_SIZE) + "\n" +
                   "  • SCP Group Size: " + std::to_string(config.SCP_GROUP_SIZE) + "\n" +
                   "\033[1;33mPerformance Metrics:\033[0m\n" +
                   "  • Total Cost: " + std::to_string(config.total_cost) + " seconds\n" +
                   "  • Compute Time: " + std::to_string(config.compute_time) + " seconds\n" +
                   "  • Communication Time: " + std::to_string(config.communication_time) + " seconds\n" +
                   "\033[1;35mConfiguration Details:\033[0m\n" +
                   "  • Deployment Strategy: " + deployment_strategy_to_string(config.deployment_strategy) + "\n" +
                   "  • Sparsity Index Groups: " + vec2d_to_string<int>(config.Reallocated_Sparsity_Idx_Per_Group) + "\n" +
                   std::string(50, '=');
        });

    // Binding the hardware configuration related classes
    py::class_<GPUSpec>(m, "GPUSpec")
        .def_readonly("name", &GPUSpec::name)
        .def_readonly("tflops_fp16", &GPUSpec::tflops_fp16)
        .def_readonly("tflops_bf16", &GPUSpec::tflops_bf16)
        .def_readonly("tflops_fp8", &GPUSpec::tflops_fp8)
        .def_readonly("memory_GB", &GPUSpec::memory_GB)
        .def_readonly("memory_bandwidth_GBps", &GPUSpec::memory_bandwidth_GBps);

    py::class_<InterconnectSpec>(m, "InterconnectSpec")
        .def_readonly("name", &InterconnectSpec::name)
        .def_readonly("intra_node_unidirectional_bandwidth_GBps", &InterconnectSpec::intra_node_unidirectional_bandwidth_GBps)
        .def_readonly("inter_node_unidirectional_bandwidth_GBps", &InterconnectSpec::inter_node_unidirectional_bandwidth_GBps)
        .def_readonly("max_gpus_per_node", &InterconnectSpec::max_gpus_per_node);

    py::class_<ServerConfig>(m, "ServerConfig")
        .def_readonly("gpu_spec", &ServerConfig::gpu_spec)
        .def_readonly("interconnect_spec", &ServerConfig::interconnect_spec)
        .def_readonly("total_gpus", &ServerConfig::total_gpus)
        .def_readonly("gpus_per_node", &ServerConfig::gpus_per_node)
        .def("get_compute_flops_per_second", &ServerConfig::get_compute_flops_per_second,
             "Get compute capability in FLOPS/s for given data type",
             py::arg("data_type"))
        .def("get_intra_node_bandwidth_in_bytes_per_second", &ServerConfig::get_intra_node_bandwidth_in_bytes_per_second,
             "Get intra-node bandwidth in bytes per second")
        .def("get_inter_node_bandwidth_in_bytes_per_second", &ServerConfig::get_inter_node_bandwidth_in_bytes_per_second,
             "Get inter-node bandwidth in bytes per second")
        .def("get_memory_per_gpu_in_bytes", &ServerConfig::get_memory_per_gpu_in_bytes,
             "Get memory per GPU in bytes")
        .def("is_valid", &ServerConfig::is_valid,
             "Validate configuration")
        .def("get_node_count", &ServerConfig::get_node_count,
             "Get number of nodes")
        .def("__str__", [](const ServerConfig& config) {
            return std::string(60, '=') + "\n" +
                   "\033[1;44m\033[1;37m SERVER CONFIGURATION \033[0m\n" +
                   std::string(60, '=') + "\n" +
                   "\033[1;36mGPU Specification:\033[0m\n" +
                   "  • Model: " + config.gpu_spec.name + "\n" +
                   "  • Memory: " + std::to_string(config.gpu_spec.memory_GB) + " GB\n" +
                   "  • Memory Bandwidth: " + std::to_string(config.gpu_spec.memory_bandwidth_GBps) + " GB/s\n" +
                   "  • FP16 Performance: " + std::to_string(config.gpu_spec.tflops_fp16) + " TFLOPS\n" +
                   "  • BF16 Performance: " + std::to_string(config.gpu_spec.tflops_bf16) + " TFLOPS\n" +
                   "  • FP8 Performance: " + std::to_string(config.gpu_spec.tflops_fp8) + " TFLOPS\n" +
                   "\033[1;33mInterconnect Specification:\033[0m\n" +
                   "  • Type: " + config.interconnect_spec.name + "\n" +
                   "  • Intra-Node Bandwidth: " + std::to_string(config.interconnect_spec.intra_node_unidirectional_bandwidth_GBps) + " GB/s\n" +
                   "  • Inter-Node Bandwidth: " + std::to_string(config.interconnect_spec.inter_node_unidirectional_bandwidth_GBps) + " GB/s\n" +
                   "  • Max GPUs per Node: " + std::to_string(config.interconnect_spec.max_gpus_per_node) + "\n" +
                   "\033[1;35mDeployment Configuration:\033[0m\n" +
                   "  • Total GPUs: " + std::to_string(config.total_gpus) + "\n" +
                   "  • GPUs per Node: " + std::to_string(config.gpus_per_node) + "\n" +
                   "  • Number of Nodes: " + std::to_string(config.get_node_count()) + "\n" +
                   "  • Total Memory: " + std::to_string(config.total_gpus * config.gpu_spec.memory_GB) + " GB\n" +
                   std::string(60, '=');
        });

    py::class_<HardwareConfigManager>(m, "HardwareConfigManager")
        .def_static("initialize", &HardwareConfigManager::initialize,
                   "Initialize the hardware configuration manager")
        .def_static("create_server_config", &HardwareConfigManager::create_server_config,
                   "Create configuration based on predefined server type",
                   py::arg("server_type"), py::arg("total_gpus"), py::arg("gpus_per_node"))
        .def_static("create_custom_config", &HardwareConfigManager::create_custom_config,
                   "Create configuration based on specific hardware specs",
                   py::arg("gpu_type"), py::arg("interconnect_type"), 
                   py::arg("total_gpus"), py::arg("gpus_per_node"))
        .def_static("create_config_with_custom_specs", &HardwareConfigManager::create_config_with_custom_specs,
                   "Create configuration with custom GPU and interconnect specs",
                   py::arg("gpu_spec"), py::arg("interconnect_spec"),
                   py::arg("gpus_per_node"), py::arg("node_count"))
        .def_static("create_config_with_user_defined_specs", &HardwareConfigManager::create_config_with_user_defined_specs,
                   "Create configuration with fully custom hardware specifications",
                   py::arg("gpu_name"), py::arg("tflops_fp16"), py::arg("tflops_bf16"), py::arg("tflops_fp8"),
                   py::arg("memory_GB"), py::arg("memory_bandwidth_GBps"),
                   py::arg("interconnect_name"), py::arg("intra_node_unidirectional_bandwidth_GBps"), 
                   py::arg("inter_node_unidirectional_bandwidth_GBps"), py::arg("max_gpus_per_node"),
                   py::arg("gpus_per_node"), py::arg("node_count"))
        .def_static("create_h100_sxm_config", &HardwareConfigManager::create_h100_sxm_config,
                   "Create SXM H100 configuration with flexible GPU count per node",
                   py::arg("total_gpus"), py::arg("gpus_per_node") = 8)
        .def_static("create_h800_sxm_config", &HardwareConfigManager::create_h800_sxm_config,
                   "Create SXM H800 configuration with flexible GPU count per node",
                   py::arg("total_gpus"), py::arg("gpus_per_node") = 8)
        .def_static("create_a100_sxm_config", &HardwareConfigManager::create_a100_sxm_config,
                   "Create SXM A100 configuration with flexible GPU count per node",
                   py::arg("total_gpus"), py::arg("gpus_per_node") = 8)
        .def_static("create_config_by_nodes", &HardwareConfigManager::create_config_by_nodes,
                   "Create configuration by specifying nodes and GPUs per node",
                   py::arg("server_type"), py::arg("nodes"), py::arg("gpus_per_node"))
        .def_static("list_predefined_servers", &HardwareConfigManager::list_predefined_servers,
                   "List all available predefined server configurations")
        .def_static("list_available_gpus", &HardwareConfigManager::list_available_gpus,
                   "List all available GPU types")
        .def_static("list_available_interconnects", &HardwareConfigManager::list_available_interconnects,
                   "List all available interconnect types")
        .def_static("print_server_info", &HardwareConfigManager::print_server_info,
                   "Print detailed server configuration information",
                   py::arg("config"))
        .def_static("print_available_options", &HardwareConfigManager::print_available_options,
                   "Print all available hardware configuration options");

    py::class_<HybridSparsityOptimizer<double>>(m, "HybridSparsityOptimizer")
        .def(py::init<const ServerConfig&, int, int, int, double, std::string, double, double, double>(),
             "Create optimizer with server configuration\n\n"
             "Args:\n"
             "    server_config: Hardware server configuration\n"
             "    HEAD_NUM: Number of attention heads\n"
             "    SEQ_LEN: Sequence length\n"
             "    HEAD_DIM: Attention head dimension\n"
             "    MEM_BUFFER_PER_GPU: Per-GPU memory buffer limit in bytes\n"
             "    DATA_TYPE: Data type ('fp16', 'bf16', or 'fp8')\n"
             "    compute_efficiency: Compute efficiency factor (0.0, 1.0], default=1.0\n"
             "                       Applied to theoretical peak compute performance\n"
             "    comm_efficiency: Communication efficiency factor (0.0, 1.0], default=1.0\n"
             "                    Applied to theoretical peak bandwidth\n"
             "    scp_overhead: SCP communication overhead in seconds, default=0.0\n"
             "                 Additional time for index computation, sparse KV setup, etc.",
             py::arg("server_config"), py::arg("HEAD_NUM"), py::arg("SEQ_LEN"),
             py::arg("HEAD_DIM"), py::arg("MEM_BUFFER_PER_GPU"), py::arg("DATA_TYPE"),
             py::arg("compute_efficiency") = 1.0, py::arg("comm_efficiency") = 1.0, py::arg("scp_overhead") = 0.0)
        .def("optimize", &HybridSparsityOptimizer<double>::optimize, 
             "Optimize parallel strategy for given sparsity configuration",
             py::arg("sparsity"), py::arg("num_threads") = 1, py::arg("verbose") = false)
        .def("get_server_config", &HybridSparsityOptimizer<double>::get_server_config,
             "Get server configuration")
        .def("get_compute_efficiency", &HybridSparsityOptimizer<double>::get_compute_efficiency,
             "Get compute efficiency factor")
        .def("get_comm_efficiency", &HybridSparsityOptimizer<double>::get_comm_efficiency,
             "Get communication efficiency factor")
        .def("get_scp_overhead", &HybridSparsityOptimizer<double>::get_scp_overhead,
             "Get SCP communication overhead in seconds")
        .def("print_server_info", &HybridSparsityOptimizer<double>::print_server_info,
             "Print server information");
} 