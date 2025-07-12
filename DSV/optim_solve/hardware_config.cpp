#include "hardware_config.h"
#include <stdexcept>
#include <iostream>
#include <iomanip>

// Static member initialization
std::unordered_map<std::string, PredefinedServerConfig> HardwareConfigManager::predefined_servers;
bool HardwareConfigManager::initialized = false;

void HardwareConfigManager::initialize_predefined_servers() {
    // SXM series servers with InfiniBand (GPU count is flexible)
    predefined_servers["H100_SXM_IB"] = PredefinedServerConfig("H100_SXM", "SXM_H100_IB", "SXM H100 with InfiniBand interconnect");
    predefined_servers["H800_SXM_IB"] = PredefinedServerConfig("H800_SXM", "SXM_H800_IB", "SXM H800 with InfiniBand interconnect");
    predefined_servers["A100_SXM_IB"] = PredefinedServerConfig("A100_SXM", "SXM_A100_IB", "SXM A100 with InfiniBand interconnect");
    
    // PCIe series servers with InfiniBand (GPU count is flexible)
    predefined_servers["H100_PCIe_IB"] = PredefinedServerConfig("H100_PCIe", "PCIe_Gen5_IB", "PCIe H100 with InfiniBand interconnect");
    predefined_servers["H800_PCIe_IB"] = PredefinedServerConfig("H800_PCIe", "PCIe_Gen5_IB", "PCIe H800 with InfiniBand interconnect");
    predefined_servers["A100_PCIe_IB"] = PredefinedServerConfig("A100_PCIe", "PCIe_Gen4_IB", "PCIe A100 with InfiniBand interconnect");
}

void HardwareConfigManager::initialize() {
    if (!initialized) {
        GPUDatabase::initialize();
        initialize_predefined_servers();
        initialized = true;
    }
}

ServerConfig HardwareConfigManager::create_server_config(const std::string& server_type, int total_gpus, int gpus_per_node) {
    initialize();
    
    auto it = predefined_servers.find(server_type);
    if (it == predefined_servers.end()) {
        throw std::invalid_argument("Unknown server type: " + server_type + 
            ". Use list_predefined_servers() to see available options.");
    }
    
    const auto& predefined = it->second;
    return create_custom_config(predefined.gpu_type, predefined.interconnect_type, total_gpus, gpus_per_node);
}

ServerConfig HardwareConfigManager::create_custom_config(const std::string& gpu_type, const std::string& interconnect_type, 
                                                         int total_gpus, int gpus_per_node) {
    initialize();
    
    const GPUSpec& gpu_spec = GPUDatabase::get_gpu_spec(gpu_type);
    const InterconnectSpec& interconnect_spec = GPUDatabase::get_interconnect_spec(interconnect_type);
    
    ServerConfig config(gpu_spec, interconnect_spec, total_gpus, gpus_per_node);
    
    if (!config.is_valid()) {
        throw std::invalid_argument(
            "Invalid server configuration: " + std::to_string(total_gpus) + " total GPUs, " +
            std::to_string(gpus_per_node) + " GPUs per node. " +
            "Check that total_gpus % gpus_per_node == 0 and " +
            "gpus_per_node <= " + std::to_string(interconnect_spec.max_gpus_per_node)
        );
    }
    
    return config;
}

ServerConfig HardwareConfigManager::create_config_with_custom_specs(
    const GPUSpec& gpu_spec,
    const InterconnectSpec& interconnect_spec,
    int gpus_per_node,
    int node_count) {
    
    int total_gpus = node_count * gpus_per_node;
    ServerConfig config(gpu_spec, interconnect_spec, total_gpus, gpus_per_node);
    
    if (!config.is_valid()) {
        throw std::invalid_argument(
            "Invalid custom server configuration: " + std::to_string(total_gpus) + " total GPUs, " +
            std::to_string(gpus_per_node) + " GPUs per node. " +
            "Check that gpus_per_node <= " + std::to_string(interconnect_spec.max_gpus_per_node)
        );
    }
    
    return config;
}

ServerConfig HardwareConfigManager::create_config_with_user_defined_specs(
    // GPU specifications
    const std::string& gpu_name,
    double tflops_fp16,
    double tflops_bf16, 
    double tflops_fp8,
    double memory_GB,
    double memory_bandwidth_GBps,
    // Interconnect specifications
    const std::string& interconnect_name,
    double intra_node_unidirectional_bandwidth_GBps,
    double inter_node_unidirectional_bandwidth_GBps,
    int max_gpus_per_node,
    // Configuration
    int gpus_per_node,
    int node_count) {
    
    // Create custom GPU specification
    GPUSpec custom_gpu_spec(gpu_name, tflops_fp16, tflops_bf16, tflops_fp8, memory_GB, memory_bandwidth_GBps);
    
    // Create custom interconnect specification
    InterconnectSpec custom_interconnect_spec(interconnect_name, 
        intra_node_unidirectional_bandwidth_GBps, 
        inter_node_unidirectional_bandwidth_GBps, 
        max_gpus_per_node);
    
    // Validate gpus_per_node against max_gpus_per_node
    if (gpus_per_node > max_gpus_per_node) {
        throw std::invalid_argument(
            "Invalid configuration: gpus_per_node (" + std::to_string(gpus_per_node) + 
            ") exceeds max_gpus_per_node (" + std::to_string(max_gpus_per_node) + ")"
        );
    }
    
    return create_config_with_custom_specs(custom_gpu_spec, custom_interconnect_spec, gpus_per_node, node_count);
}

ServerConfig HardwareConfigManager::create_h100_sxm_config(int total_gpus, int gpus_per_node) {
    return create_custom_config("H100_SXM", "SXM_H100_IB", total_gpus, gpus_per_node);
}

ServerConfig HardwareConfigManager::create_h800_sxm_config(int total_gpus, int gpus_per_node) {
    return create_custom_config("H800_SXM", "SXM_H800_IB", total_gpus, gpus_per_node);
}

ServerConfig HardwareConfigManager::create_a100_sxm_config(int total_gpus, int gpus_per_node) {
    return create_custom_config("A100_SXM", "SXM_A100_IB", total_gpus, gpus_per_node);
}

ServerConfig HardwareConfigManager::create_config_by_nodes(const std::string& server_type, int nodes, int gpus_per_node) {
    int total_gpus = nodes * gpus_per_node;
    return create_server_config(server_type, total_gpus, gpus_per_node);
}

std::vector<std::string> HardwareConfigManager::list_predefined_servers() {
    initialize();
    std::vector<std::string> server_list;
    for (const auto& pair : predefined_servers) {
        server_list.push_back(pair.first);
    }
    return server_list;
}

std::vector<std::string> HardwareConfigManager::list_available_gpus() {
    return GPUDatabase::list_available_gpus();
}

std::vector<std::string> HardwareConfigManager::list_available_interconnects() {
    return GPUDatabase::list_available_interconnects();
}

void HardwareConfigManager::print_server_info(const ServerConfig& config) {
    std::cout << "\n=== Server Configuration ===" << std::endl;
    std::cout << "GPU Model: " << config.gpu_spec.name << std::endl;
    std::cout << "Interconnect: " << config.interconnect_spec.name << std::endl;
    std::cout << "Total GPUs: " << config.total_gpus << std::endl;
    std::cout << "GPUs per Node: " << config.gpus_per_node << std::endl;
    std::cout << "Node Count: " << config.get_node_count() << std::endl;
    
    std::cout << "\n--- GPU Performance ---" << std::endl;
    std::cout << "FP16 TFLOPS: " << config.gpu_spec.tflops_fp16 << std::endl;
    std::cout << "BF16 TFLOPS: " << config.gpu_spec.tflops_bf16 << std::endl;
    std::cout << "FP8 TFLOPS: " << config.gpu_spec.tflops_fp8 << std::endl;
    std::cout << "Memory: " << config.gpu_spec.memory_GB << " GB" << std::endl;
    std::cout << "Memory Bandwidth: " << config.gpu_spec.memory_bandwidth_GBps << " GB/s" << std::endl;
    
    std::cout << "\n--- Network Performance ---" << std::endl;
    std::cout << "Intra-node Bandwidth: " << config.interconnect_spec.intra_node_unidirectional_bandwidth_GBps << " GB/s (unidirectional)" << std::endl;
    std::cout << "Inter-node Bandwidth: " << config.interconnect_spec.inter_node_unidirectional_bandwidth_GBps << " GB/s (unidirectional)" << std::endl;
    std::cout << "Max GPUs per Node: " << config.interconnect_spec.max_gpus_per_node << std::endl;
    
    std::cout << "\n--- Configuration Status ---" << std::endl;
    std::cout << "Valid: " << (config.is_valid() ? "✅ Yes" : "❌ No") << std::endl;
    std::cout << "========================" << std::endl;
}

void HardwareConfigManager::print_available_options() {
    initialize();
    
    std::cout << "\n=== Available Hardware Options ===" << std::endl;
    
    std::cout << "\n--- Predefined Server Types ---" << std::endl;
    auto servers = list_predefined_servers();
    for (const auto& server_name : servers) {
        const auto& predefined = predefined_servers[server_name];
        std::cout << std::setw(15) << server_name 
                  << " (GPU: " << std::setw(12) << predefined.gpu_type 
                  << ", Network: " << std::setw(15) << predefined.interconnect_type << ")" << std::endl;
        std::cout << std::setw(15) << "" << " - " << predefined.description << std::endl;
    }
    
    std::cout << "\n--- Available GPU Types ---" << std::endl;
    auto gpus = list_available_gpus();
    for (size_t i = 0; i < gpus.size(); ++i) {
        std::cout << std::setw(15) << gpus[i];
        if ((i + 1) % 3 == 0) std::cout << std::endl;
    }
    if (gpus.size() % 3 != 0) std::cout << std::endl;
    
    std::cout << "\n--- Available Interconnect Types ---" << std::endl;
    auto interconnects = list_available_interconnects();
    for (size_t i = 0; i < interconnects.size(); ++i) {
        std::cout << std::setw(18) << interconnects[i];
        if ((i + 1) % 3 == 0) std::cout << std::endl;
    }
    if (interconnects.size() % 3 != 0) std::cout << std::endl;
    
    std::cout << "\nUsage Examples:" << std::endl;
    std::cout << "  // Flexible GPU count per node:" << std::endl;
    std::cout << "  auto config = HardwareConfigManager::create_server_config(\"H100_SXM_IB\", 32, 4);  // 4 GPUs/node" << std::endl;
    std::cout << "  auto config = HardwareConfigManager::create_server_config(\"H800_SXM_IB\", 64, 8);  // 8 GPUs/node" << std::endl;
    std::cout << "  // Quick methods with default 8 GPUs/node:" << std::endl;
    std::cout << "  auto config = HardwareConfigManager::create_h100_sxm_config(64);       // 64 GPUs, 8/node" << std::endl;
    std::cout << "  auto config = HardwareConfigManager::create_h800_sxm_config(32, 4);    // 32 GPUs, 4/node" << std::endl;
    std::cout << "  // Specify by nodes:" << std::endl;
    std::cout << "  auto config = HardwareConfigManager::create_config_by_nodes(\"H100_SXM_IB\", 8, 4);  // 8 nodes, 4 GPUs each" << std::endl;
    std::cout << "  // Custom hardware specs:" << std::endl;
    std::cout << "  auto config = HardwareConfigManager::create_config_with_user_defined_specs(...);  // Fully custom" << std::endl;
    std::cout << "========================" << std::endl;
} 