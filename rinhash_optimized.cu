// rinhash_optimized.cu - High-Performance Mining Implementation
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>

// Include your device headers
#include "rinhash_device.cuh"
#include "argon2d_device.cuh"
#include "blake3_device.cuh"
#include "sha3-256.cu"

// Global memory for Argon2d workspace (pre-allocated)
__device__ uint8_t* g_argon2d_memory = nullptr;
__device__ size_t g_memory_per_thread = 0;

// Optimized RinHash kernel for parallel batch processing
__global__ void rinhash_mining_kernel(
    const uint8_t* __restrict__ d_headers,     // Input: 80-byte headers
    uint8_t* __restrict__ d_outputs,           // Output: 32-byte hashes
    uint32_t num_blocks,
    uint32_t m_cost_kb
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_blocks) return;
    
    // Thread-local storage for intermediate results
    uint8_t blake3_result[32];
    uint8_t argon2d_result[32];
    uint8_t final_hash[32];
    
    // Get pointers for this thread's data
    const uint8_t* header = d_headers + tid * 80;
    uint8_t* output = d_outputs + tid * 32;
    
    // Step 1: BLAKE3 hash using your device function
    light_hash_device(header, 80, blake3_result);
    
    // Step 2: Argon2d hash
    // Each thread gets its own memory segment
    size_t memory_offset = tid * g_memory_per_thread;
    uint8_t* thread_memory = g_argon2d_memory + memory_offset;
    
    // RinCoin salt (as per your original)
    uint8_t salt[11] = {'R','i','n','C','o','i','n','S','a','l','t'};
    
    // Call your Argon2d device function
    device_argon2d_hash(
        argon2d_result,     // output
        blake3_result,      // input
        32,                 // input length
        2,                  // t_cost (iterations)
        m_cost_kb,          // m_cost (memory in KB)
        1,                  // lanes
        (block*)thread_memory,  // pre-allocated memory
        salt,               // salt
        11                  // salt length
    );
    
    // Step 3: SHA3-256 final hash
    sha3_256_device(argon2d_result, 32, final_hash);
    
    // Store result
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        output[i] = final_hash[i];
    }
}

// Memory initialization kernel
__global__ void init_argon2d_memory(uint8_t* memory, size_t total_size) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_threads = gridDim.x * blockDim.x;
    
    // Initialize memory in parallel
    for (size_t i = tid; i < total_size; i += total_threads) {
        memory[i] = 0;
    }
}

// High-performance batch processing function
extern "C" cudaError_t rinhash_cuda_batch_optimized(
    const uint8_t* d_headers,
    uint8_t* d_outputs, 
    uint32_t num_blocks,
    uint32_t m_cost_kb
) {
    cudaError_t err;
    
    // Calculate memory requirements
    size_t memory_per_thread = m_cost_kb * 1024; // Argon2d memory per thread
    size_t total_memory = num_blocks * memory_per_thread;
    
    // Allocate global Argon2d memory if not already done
    static uint8_t* d_argon2d_memory = nullptr;
    static size_t allocated_memory = 0;
    
    if (d_argon2d_memory == nullptr || allocated_memory < total_memory) {
        if (d_argon2d_memory) {
            cudaFree(d_argon2d_memory);
        }
        
        err = cudaMalloc(&d_argon2d_memory, total_memory);
        if (err != cudaSuccess) {
            return err;
        }
        
        allocated_memory = total_memory;
        
        // Initialize memory
        int init_blocks = (total_memory + 255) / 256;
        init_argon2d_memory<<<init_blocks, 256>>>(d_argon2d_memory, total_memory);
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            return err;
        }
        
        // Set device pointers
        err = cudaMemcpyToSymbol(g_argon2d_memory, &d_argon2d_memory, sizeof(uint8_t*));
        if (err != cudaSuccess) return err;
        
        err = cudaMemcpyToSymbol(g_memory_per_thread, &memory_per_thread, sizeof(size_t));
        if (err != cudaSuccess) return err;
    }
    
    // Launch configuration
    int threads_per_block = 256;
    int blocks = (num_blocks + threads_per_block - 1) / threads_per_block;
    
    // Launch the optimized mining kernel
    rinhash_mining_kernel<<<blocks, threads_per_block>>>(
        d_headers,
        d_outputs, 
        num_blocks,
        m_cost_kb
    );
    
    return cudaGetLastError();
}

// Wrapper for the main.c integration
extern "C" cudaError_t rinhash_full_pipeline(
    const uint8_t* d_headers,
    uint8_t* d_final_hashes,
    uint32_t num_items,
    uint32_t m_cost_kb
) {
    return rinhash_cuda_batch_optimized(d_headers, d_final_hashes, num_items, m_cost_kb);
}

// Host-side batch function (for compatibility with your original interface)
extern "C" void rinhash_cuda_batch(
    const uint8_t* block_headers,
    size_t block_header_len,
    uint8_t* outputs,
    uint32_t num_blocks
) {
    if (block_header_len != 80) {
        fprintf(stderr, "Error: Expected 80-byte headers, got %zu bytes\n", block_header_len);
        return;
    }
    
    cudaError_t err;
    
    // Allocate device memory
    uint8_t *d_headers, *d_outputs;
    size_t headers_size = num_blocks * 80;
    size_t outputs_size = num_blocks * 32;
    
    err = cudaMalloc(&d_headers, headers_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate headers: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc(&d_outputs, outputs_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate outputs: %s\n", cudaGetErrorString(err));
        cudaFree(d_headers);
        return;
    }
    
    // Copy input data
    err = cudaMemcpy(d_headers, block_headers, headers_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy headers: %s\n", cudaGetErrorString(err));
        cudaFree(d_headers);
        cudaFree(d_outputs);
        return;
    }
    
    // Run optimized batch processing
    err = rinhash_cuda_batch_optimized(d_headers, d_outputs, num_blocks, 64);
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_headers);
        cudaFree(d_outputs);
        return;
    }
    
    // Synchronize and copy results
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Synchronization failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_headers);
        cudaFree(d_outputs);
        return;
    }
    
    err = cudaMemcpy(outputs, d_outputs, outputs_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy outputs: %s\n", cudaGetErrorString(err));
    }
    
    // Cleanup
    cudaFree(d_headers);
    cudaFree(d_outputs);
}

// Cleanup function
extern "C" void rinhash_cuda_cleanup() {
    uint8_t* d_memory = nullptr;
    cudaMemcpyFromSymbol(&d_memory, g_argon2d_memory, sizeof(uint8_t*));
    if (d_memory) {
        cudaFree(d_memory);
        d_memory = nullptr;
        cudaMemcpyToSymbol(g_argon2d_memory, &d_memory, sizeof(uint8_t*));
    }
}
