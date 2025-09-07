#ifndef BLAKE3_DEVICE_CUH
#define BLAKE3_DEVICE_CUH

#include <stdint.h>
#include <cuda_runtime.h>

__device__ void light_hash_device(const uint8_t* input, size_t input_len, uint8_t* output) {
    // Simple Blake3 placeholder - ersetzt durch echte Implementation
    for (int i = 0; i < 32; i++) {
        output[i] = input[i % input_len] ^ (uint8_t)(i * 0x5A);
    }
}

#endif
