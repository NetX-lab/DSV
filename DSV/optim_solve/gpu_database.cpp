#include "gpu_database.h"
#include <stdexcept>
#include <vector>

// Static member initialization
std::unordered_map<std::string, GPUSpec> GPUDatabase::gpu_specs;
std::unordered_map<std::string, InterconnectSpec> GPUDatabase::interconnect_specs;
bool GPUDatabase::initialized = false;

void GPUDatabase::initialize_gpu_specs() {
    // NVIDIA H100 series
    // double tflops_fp16;          // FP16 TFLOPS
    // double tflops_bf16;          // BF16 TFLOPS  
    // double tflops_fp8;           // FP8 TFLOPS
    // double memory_gb;            // GPU memory capacity (GB)
    // double memory_bandwidth_GBps; // GPU memory bandwidth (GB/s)
    gpu_specs["H100_SXM"] = GPUSpec("H100_SXM", 1979.0, 1979.0, 3958.0, 80.0, 3352.0);
    gpu_specs["H100_PCIe"] = GPUSpec("H100_PCIe", 1513.0, 1513.0, 3026.0, 80.0, 2000.0);
    
    // NVIDIA H800 series (China special edition)
    gpu_specs["H800_SXM"] = GPUSpec("H800_SXM", 1979.0, 1979.0, 3958.0, 80.0, 3352.0);
    gpu_specs["H800_PCIe"] = GPUSpec("H800_PCIe", 1513.0, 1513.0, 3026.0, 80.0, 2000.0);
    
    // NVIDIA A100 series
    gpu_specs["A100_SXM"] = GPUSpec("A100_SXM", 312.0, 312.0, 0.0, 80.0, 2039.0);
    gpu_specs["A100_PCIe"] = GPUSpec("A100_PCIe", 312.0, 312.0, 0.0, 80.0, 1935.0);
}

void GPUDatabase::initialize_interconnect_specs() {
    // SXM topology with different inter-node connections
    // double intra_node_unidirectional_bandwidth_GBps;  // Intra-node unidirectional bandwidth (GB/s)
    // double inter_node_unidirectional_bandwidth_GBps;  // Inter-node unidirectional bandwidth (GB/s)
    // int max_gpus_per_node;  
    interconnect_specs["SXM_H100_IB"] = InterconnectSpec("SXM_H100_IB", 450.0, 200.0/8, 8); // NVLink 4.0 unidirectional, InfiniBand HDR
    interconnect_specs["SXM_H800_IB"] = InterconnectSpec("SXM_H800_IB", 450.0, 200.0/8, 8); // NVLink 4.0 unidirectional, restricted inter-node
    interconnect_specs["SXM_A100_IB"] = InterconnectSpec("SXM_A100_IB", 300.0, 200.0/8, 8); // NVLink 3.0 unidirectional, InfiniBand HDR
    
    // PCIe topology
    interconnect_specs["PCIe_Gen4_IB"] = InterconnectSpec("PCIe_Gen4_IB", 32.0, 200.0/8, 8);  // PCIe 4.0 x16 unidirectional, InfiniBand
    interconnect_specs["PCIe_Gen5_IB"] = InterconnectSpec("PCIe_Gen5_IB", 64.0, 200.0/8, 8); // PCIe 5.0 x16 unidirectional, InfiniBand HDR
}

void GPUDatabase::initialize() {
    if (!initialized) {
        initialize_gpu_specs();
        initialize_interconnect_specs();
        initialized = true;
    }
}

const GPUSpec& GPUDatabase::get_gpu_spec(const std::string& gpu_name) {
    initialize();
    auto it = gpu_specs.find(gpu_name);
    if (it == gpu_specs.end()) {
        throw std::invalid_argument("Unknown GPU type: " + gpu_name + 
            ". Use list_available_gpus() to see available options.");
    }
    return it->second;
}

const InterconnectSpec& GPUDatabase::get_interconnect_spec(const std::string& interconnect_name) {
    initialize();
    auto it = interconnect_specs.find(interconnect_name);
    if (it == interconnect_specs.end()) {
        throw std::invalid_argument("Unknown interconnect type: " + interconnect_name +
            ". Use list_available_interconnects() to see available options.");
    }
    return it->second;
}

std::vector<std::string> GPUDatabase::list_available_gpus() {
    initialize();
    std::vector<std::string> gpu_list;
    for (const auto& pair : gpu_specs) {
        gpu_list.push_back(pair.first);
    }
    return gpu_list;
}

std::vector<std::string> GPUDatabase::list_available_interconnects() {
    initialize();
    std::vector<std::string> interconnect_list;
    for (const auto& pair : interconnect_specs) {
        interconnect_list.push_back(pair.first);
    }
    return interconnect_list;
}

bool GPUDatabase::has_gpu(const std::string& gpu_name) {
    initialize();
    return gpu_specs.find(gpu_name) != gpu_specs.end();
}

bool GPUDatabase::has_interconnect(const std::string& interconnect_name) {
    initialize();
    return interconnect_specs.find(interconnect_name) != interconnect_specs.end();
}

void GPUDatabase::add_custom_gpu(const GPUSpec& spec) {
    initialize();
    gpu_specs[spec.name] = spec;
}

void GPUDatabase::add_custom_interconnect(const InterconnectSpec& spec) {
    initialize();
    interconnect_specs[spec.name] = spec;
} 