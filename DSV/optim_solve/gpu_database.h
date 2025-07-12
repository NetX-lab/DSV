#pragma once
#include <string>
#include <vector>
#include <unordered_map>

// GPU hardware specification structure
struct GPUSpec {
    std::string name;
    double tflops_fp16;          // FP16 TFLOPS
    double tflops_bf16;          // BF16 TFLOPS  
    double tflops_fp8;           // FP8 TFLOPS
    double memory_GB;            // GPU memory capacity (GB)
    double memory_bandwidth_GBps; // GPU memory bandwidth (GB/s)
    
    GPUSpec() = default;
    GPUSpec(const std::string& n, double fp16, double bf16, double fp8, double mem, double mem_bw)
        : name(n), tflops_fp16(fp16), tflops_bf16(bf16), tflops_fp8(fp8), 
          memory_GB(mem), memory_bandwidth_GBps(mem_bw) {}
};

// Interconnect specification structure
struct InterconnectSpec {
    std::string name;
    double intra_node_unidirectional_bandwidth_GBps;  // Intra-node unidirectional bandwidth (GB/s)
    double inter_node_unidirectional_bandwidth_GBps;  // Inter-node unidirectional bandwidth (GB/s)
    int max_gpus_per_node;                             // Maximum GPUs per node
    
    InterconnectSpec() = default;
    InterconnectSpec(const std::string& n, double intra, double inter, int max_gpu)
        : name(n), intra_node_unidirectional_bandwidth_GBps(intra), inter_node_unidirectional_bandwidth_GBps(inter), 
          max_gpus_per_node(max_gpu) {}
};

// GPU database class
class GPUDatabase {
private:
    static std::unordered_map<std::string, GPUSpec> gpu_specs;
    static std::unordered_map<std::string, InterconnectSpec> interconnect_specs;
    static bool initialized;
    
    static void initialize_gpu_specs();
    static void initialize_interconnect_specs();
    
public:
    static void initialize();
    
    static const GPUSpec& get_gpu_spec(const std::string& gpu_name);
    static const InterconnectSpec& get_interconnect_spec(const std::string& interconnect_name);
    
    static std::vector<std::string> list_available_gpus();
    static std::vector<std::string> list_available_interconnects();
    
    static bool has_gpu(const std::string& gpu_name);
    static bool has_interconnect(const std::string& interconnect_name);
    
    // Add custom GPU/interconnect specifications
    static void add_custom_gpu(const GPUSpec& spec);
    static void add_custom_interconnect(const InterconnectSpec& spec);
}; 