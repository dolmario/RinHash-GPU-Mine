// rinhash_optimized.cu – korrigiert: keine Device-Globals, 64-bit Offsets, sauberer Sync
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include "rinhash_params.h"
#include "argon2d_device.cuh"
#include "blake3_device.cuh"
#include "sha3_256_device.cuh"  // nur Prototyp, .cu wird separat kompiliert

static uint8_t* g_d_argon2d_memory_host = nullptr;
static size_t   g_allocated_memory = 0;

__global__ void init_argon2d_memory(uint8_t* memory, size_t total_size){
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    for (size_t i = tid; i < total_size; i += total_threads) memory[i] = 0;
}

__global__ void rinhash_mining_kernel(
    const uint8_t* __restrict__ d_headers,
    uint8_t*       __restrict__ d_outputs,
    uint32_t num_items,
    uint8_t* __restrict__ argon_pool,
    size_t   memory_per_thread,
    uint32_t m_cost_kb,
    uint32_t t_cost,
    uint32_t lanes
){
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) return;
    if (!argon_pool || memory_per_thread == 0) return;

    uint8_t blake3_result[32];
    uint8_t argon2d_result[32];
    uint8_t sha3_in[136];
    uint8_t final_hash[32];

    const uint8_t* header = d_headers + (size_t)tid * 80;
    uint8_t*       out    = d_outputs + (size_t)tid * 32;

    light_hash_device(header, 80, blake3_result);

    const size_t memory_offset = (size_t)tid * memory_per_thread;
    uint8_t* thread_memory     = argon_pool + memory_offset;
    const uint8_t salt[11] = {'R','i','n','C','o','i','n','S','a','l','t'};

    device_argon2d_hash(
        /*out*/    argon2d_result,
        /*in*/     blake3_result,
        /*inLen*/  32,
        /*t_cost*/ t_cost,
        /*m_kib*/  m_cost_kb,
        /*lanes*/  lanes,
        /*mem*/    (block*)thread_memory,
        /*salt*/   (const uint8_t*)salt,
        /*saltLen*/11
    );

    // SHA3 sicher: 136-Byte Zero-Pad-Buffer, 32B Input vorne


    #pragma unroll


    for (int i=0;i<136;i++) sha3_in[i]=0;


    #pragma unroll


    for (int i=0;i<32;i++)  sha3_in[i]=argon2d_result[i];


    sha3_256_device(sha3_in, 32, final_hash);

    #pragma unroll
    for (int i = 0; i < 32; ++i) out[i] = final_hash[i];
}

extern "C" cudaError_t rinhash_cuda_batch_optimized(
    const uint8_t* d_headers,
    uint8_t*       d_outputs,
    uint32_t       num_items,
    uint32_t       m_cost_kb
){
    if (num_items == 0) return cudaSuccess;

    const size_t memory_per_thread = (size_t)m_cost_kb * 1024ull;
    const size_t total_memory      = (size_t)num_items * memory_per_thread;

    cudaError_t err;

    if (g_d_argon2d_memory_host == nullptr || g_allocated_memory < total_memory) {
        if (g_d_argon2d_memory_host) cudaFree(g_d_argon2d_memory_host);
        g_d_argon2d_memory_host = nullptr;
        g_allocated_memory      = 0;

        err = cudaMalloc(&g_d_argon2d_memory_host, total_memory);
        if (err != cudaSuccess) return err;
        g_allocated_memory = total_memory;

        const int tpb = 256;
        size_t blocks64 = (total_memory + tpb - 1) / tpb;
        int init_blocks = (int)((blocks64 > 65535) ? 65535 : blocks64);
        init_argon2d_memory<<<init_blocks, tpb>>>(g_d_argon2d_memory_host, total_memory);
        err = cudaGetLastError();      if (err != cudaSuccess) return err;
        err = cudaDeviceSynchronize(); if (err != cudaSuccess) return err;
    }

    const int tpb    = 256;
    const int blocks = (num_items + tpb - 1) / tpb;

    rinhash_mining_kernel<<<blocks, tpb>>>(
        d_headers, d_outputs, num_items,
        g_d_argon2d_memory_host, memory_per_thread,
        m_cost_kb, RIN_ARGON2_T, RIN_ARGON2_LANES
    );

    err = cudaGetLastError();      if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

extern "C" cudaError_t rinhash_full_pipeline(
    const uint8_t* d_headers,
    uint8_t*       d_final_hashes,
    uint32_t       num_items,
    uint32_t       m_cost_kb
){
    return rinhash_cuda_batch_optimized(d_headers, d_final_hashes, num_items, m_cost_kb);
}

extern "C" void rinhash_cuda_batch(
    const uint8_t* block_headers,
    size_t         block_header_len,
    uint8_t*       outputs,
    uint32_t       num_blocks
){
    if (block_header_len != 80) {
        fprintf(stderr, "Error: Expected 80-byte headers, got %zu bytes\n", block_header_len);
        return;
    }
    uint8_t *d_headers = nullptr, *d_outputs = nullptr;
    const size_t headers_size = (size_t)num_blocks * 80;
    const size_t outputs_size = (size_t)num_blocks * 32;

    cudaError_t err;
    err = cudaMalloc(&d_headers, headers_size);
    if (err != cudaSuccess) { fprintf(stderr, "Failed to allocate headers: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&d_outputs, outputs_size);
    if (err != cudaSuccess) { fprintf(stderr, "Failed to allocate outputs: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(d_headers, block_headers, headers_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "Failed to copy headers: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = rinhash_cuda_batch_optimized(d_headers, d_outputs, num_blocks, RIN_ARGON2_M_KIB);
    if (err != cudaSuccess) { fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(outputs, d_outputs, outputs_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "Failed to copy outputs: %s\n", cudaGetErrorString(err)); }

cleanup:
    if (d_headers) cudaFree(d_headers);
    if (d_outputs) cudaFree(d_outputs);
}

extern "C" void rinhash_cuda_cleanup(){
    if (g_d_argon2d_memory_host){
        cudaFree(g_d_argon2d_memory_host);
        g_d_argon2d_memory_host = nullptr;
        g_allocated_memory = 0;
    }
}