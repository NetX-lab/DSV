#pragma once
#include "gpu_database.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

// Server configuration structure
struct ServerConfig {
    GPUSpec gpu_spec;
    InterconnectSpec interconnect_spec;
    int total_gpus;
    int gpus_per_node;
    
    ServerConfig() = default;
    ServerConfig(const GPUSpec& gpu, const InterconnectSpec& interconnect, int total, int per_node)
        : gpu_spec(gpu), interconnect_spec(interconnect), total_gpus(total), gpus_per_node(per_node) {}
        
    // Get TFLOPS based on data type (convert to FLOPS/s)
    double get_compute_flops_per_second(const std::string& data_type) const {
        double tflops = 0.0;
        if (data_type == "fp16") tflops = gpu_spec.tflops_fp16;
        else if (data_type == "bf16") tflops = gpu_spec.tflops_bf16;
        else if (data_type == "fp8") tflops = gpu_spec.tflops_fp8;
        else throw std::invalid_argument("Unsupported data type: " + data_type);
        
        return tflops * 1e12; // Convert to FLOPS/s
    }
    
    // Get intra-node bandwidth in bytes per second
    double get_intra_node_bandwidth_in_bytes_per_second() const {
        return interconnect_spec.intra_node_unidirectional_bandwidth_GBps * 1e9;
    }
    
    // Get inter-node bandwidth in bytes per second
    double get_inter_node_bandwidth_in_bytes_per_second() const {
        return interconnect_spec.inter_node_unidirectional_bandwidth_GBps * 1e9;
    }
    
    // Get memory per GPU in bytes
    double get_memory_per_gpu_in_bytes() const {
        return gpu_spec.memory_GB * 1e9;
    }
    
    // Validate configuration
    bool is_valid() const {
        return total_gpus > 0 && 
               gpus_per_node > 0 && 
               gpus_per_node <= interconnect_spec.max_gpus_per_node &&
               total_gpus % gpus_per_node == 0;
    }
    
    // Get node count
    int get_node_count() const {
        return total_gpus / gpus_per_node;
    }
};

// Predefined server configuration mapping (only stores hardware types, not GPU count)
struct PredefinedServerConfig {
    std::string gpu_type;
    std::string interconnect_type;
    std::string description;
    
    PredefinedServerConfig() = default;
    PredefinedServerConfig(const std::string& gpu, const std::string& interconnect, const std::string& desc)
        : gpu_type(gpu), interconnect_type(interconnect), description(desc) {}
};

// Hardware configuration manager
class HardwareConfigManager {
private:
    static std::unordered_map<std::string, PredefinedServerConfig> predefined_servers;
    static bool initialized;
    
    static void initialize_predefined_servers();
    
public:
    static void initialize();
    
    // Create configuration based on predefined server type (must specify GPUs per node)
    static ServerConfig create_server_config(
        const std::string& server_type,
        int total_gpus,
        int gpus_per_node
    );
    
    // Create configuration based on specific hardware specs
    static ServerConfig create_custom_config(
        const std::string& gpu_type,
        const std::string& interconnect_type, 
        int total_gpus,
        int gpus_per_node
    );
    
    // Create configuration with custom GPU and interconnect specs (using existing spec objects)
    static ServerConfig create_config_with_custom_specs(
        const GPUSpec& gpu_spec,
        const InterconnectSpec& interconnect_spec,
        int gpus_per_node,
        int node_count
    );
    
    // Create configuration with fully custom hardware specifications (user-defined specs)
    static ServerConfig create_config_with_user_defined_specs(
        // GPU specifications
        const std::string& gpu_name,
        double tflops_fp16,
        double tflops_bf16, 
        double tflops_fp8,
        double memory_gb,
        double memory_bandwidth_GBps,
        // Interconnect specifications
        const std::string& interconnect_name,
        double intra_node_unidirectional_bandwidth_GBps,
        double inter_node_unidirectional_bandwidth_GBps,
        int max_gpus_per_node,
        // Configuration
        int gpus_per_node,
        int node_count
    );
    
    // Quick create common SXM configurations with flexible GPU count per node
    static ServerConfig create_h100_sxm_config(int total_gpus, int gpus_per_node = 8);
    static ServerConfig create_h800_sxm_config(int total_gpus, int gpus_per_node = 8);
    static ServerConfig create_a100_sxm_config(int total_gpus, int gpus_per_node = 8);
    
    // Create configuration by specifying nodes and GPUs per node
    static ServerConfig create_config_by_nodes(const std::string& server_type, int nodes, int gpus_per_node);
    
    // List all available options
    static std::vector<std::string> list_predefined_servers();
    static std::vector<std::string> list_available_gpus();
    static std::vector<std::string> list_available_interconnects();
    
    // Print configuration information
    static void print_server_info(const ServerConfig& config);
    static void print_available_options();
}; 