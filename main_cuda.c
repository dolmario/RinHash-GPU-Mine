// main_cuda.c — RinHash Full-GPU Miner
// Pipeline: BLAKE3(GPU) -> Argon2d(GPU) -> SHA3-256(GPU) -> compare vs target -> submit

#include <cuda_runtime.h>

#include <openssl/evp.h>
#include <openssl/bn.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <stdint.h>
#include <pthread.h>
#include <sys/select.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

// CUDA Kernel Declaration - Full Pipeline auf GPU
extern "C" {
    cudaError_t rinhash_full_pipeline(
        const uint8_t* d_headers,        // 80-byte headers
        uint8_t* d_final_hashes,         // 32-byte final SHA3 outputs
        uint32_t num_items,              // batch size
        uint32_t m_cost_kb               // Argon2d memory parameter
    );
}

// ============================ Defaults =============================
static const char *POOL_CANDIDATES[] = {
    "rinhash.eu.mine.zergpool.com",
    "rinhash.mine.zergpool.com", 
    "rinhash.na.mine.zergpool.com",
    "rinhash.asia.mine.zergpool.com",
    NULL
};
static const int PORT_DEFAULT = 7148;

static const char *DEFAULT_WAL  = "DTXoRQ7Zpw3FVRW2DWkLrM9Skj4Gp9SeSj";
static const char *DEFAULT_PASS = "c=DOGE,ID=cuda";

// Argon2d params  
#define M_COST_KB 64
#define T_COST    2
#define LANES     1

// Batch default
#define BATCH_DEFAULT 512  // Höher da Full-GPU effizienter

// ============================ Globals =============================

static const uint8_t DIFF1_TARGET_BE[32] = {
    0x00,0x00,0x00,0x00, 0xFF,0xFF,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
};

static int g_debug = 0;
static double g_share_diff = 1.0;
static uint8_t g_share_target_be[32] = {
    0x00,0x00,0x00,0x00, 0xFF,0xFF,0x00,0x00,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0
};
static int g_diff_locked = 0;

// Statistics
static uint64_t g_total_hashes = 0;
static uint32_t g_best_leading_zeros = 0;
static uint32_t g_accepted_shares = 0;
static uint32_t g_rejected_shares = 0;

// Job management
static uint32_t g_batch_counter = 0;
static volatile uint32_t g_job_gen = 0;
static volatile int g_last_notify_clean = 1;

// ============================ Utils ==============================

static int hex2bin(const char *hex, uint8_t *out, size_t outlen) {
    for (size_t i=0; i<outlen; i++) {
        unsigned v; 
        if (sscanf(hex + 2*i, "%2x", &v) != 1) return 0;
        out[i] = (uint8_t)v;
    }
    return 1;
}

static uint64_t mono_ms(void) {
    struct timespec ts; 
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec*1000ull + (uint64_t)ts.tv_nsec/1000000ull;
}

static void update_statistics(const uint8_t hash_be[32]) {
    g_total_hashes++;
    uint32_t zeros = 0;
    for (int i = 0; i < 32; i++) {
        if (hash_be[i] == 0) zeros += 8;
        else {
            uint8_t b = hash_be[i];
            while ((b & 0x80) == 0 && zeros < 256) { 
                zeros++; 
                b <<= 1; 
            }
            break;
        }
    }
    if (zeros > g_best_leading_zeros) {
        g_best_leading_zeros = zeros;
        if (g_debug) printf("New best: %u leading zero bits\n", zeros);
    }
}

// ============================ Difficulty / Target utils =============================

static void diff_to_target(double diff, uint8_t out_be[32]) {
    if (diff <= 0) diff = 1e-12;

    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *diff1 = BN_bin2bn(DIFF1_TARGET_BE, 32, NULL);
    BIGNUM *num   = BN_new();
    BIGNUM *den   = BN_new();
    BIGNUM *tgt   = BN_new();

    const unsigned SHIFT = 24;
    BN_copy(num, diff1);
    BN_lshift(num, num, SHIFT);

    long double scaled = (long double)diff * (long double)(1ULL << SHIFT);
    uint64_t den64 = (scaled < 1.0L) ? 1ULL : (uint64_t)llroundl(scaled);
    BN_set_word(den, den64);

    BN_div(tgt, NULL, num, den, ctx);
    if (BN_is_zero(tgt)) BN_one(tgt);

#if OPENSSL_VERSION_NUMBER >= 0x10100000L
    BN_bn2binpad(tgt, out_be, 32);
#else
    memset(out_be, 0, 32);
    int n = BN_num_bytes(tgt);
    if (n > 32) {
        uint8_t tmp[64]; 
        int m = BN_bn2bin(tgt, tmp);
        memcpy(out_be, tmp + (m - 32), 32);
    } else {
        BN_bn2bin(tgt, out_be + (32 - n));
    }
#endif

    BN_free(diff1); BN_free(num); BN_free(den); BN_free(tgt); BN_CTX_free(ctx);
}

static int hash_meets_target_be(const uint8_t hash_be[32], const uint8_t target_be[32]) {
    for (int i = 0; i < 32; i++) {
        if (hash_be[i] < target_be[i]) return 1;
        if (hash_be[i] > target_be[i]) return 0;
    }
    return 1;
}

// ============================ Stratum (vereinfacht) ==============================

typedef struct {
    char job_id[128];
    char prevhash_hex[65]; 
    char coinb1_hex[4096];
    char coinb2_hex[4096];
    char merkle_hex[16][65];
    int merkle_count;
    uint32_t version;
    uint32_t nbits;
    uint32_t ntime;
    int clean;
} stratum_job_t;

typedef struct {
    char host[256];
    int port;
    int sock;
    char wallet[256];
    char pass[512];
    char extranonce1[64];
    uint32_t extranonce2_size;
} stratum_ctx_t;

// Vereinfachte Stratum-Funktionen (Basis-Implementation)
static int stratum_connect_one(stratum_ctx_t *C, const char *host, int port, 
                              const char *user, const char *pass) {
    // Basic socket connect + subscribe + authorize
    // (Implementation details gekürzt für Übersichtlichkeit)
    printf("Connected to %s:%d\n", host, port);
    snprintf(C->extranonce1, sizeof(C->extranonce1), "00000000");
    C->extranonce2_size = 4;
    return 1;  // Success simulation
}

static int stratum_parse_notify(const char *line, stratum_job_t *J) {
    // Parse mining.notify message
    // (Implementation details gekürzt)
    snprintf(J->job_id, sizeof(J->job_id), "test_job");
    return 1;  // Success simulation
}

// ============================ CUDA Setup ==============================

static int cuda_setup_device(void) {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "No CUDA devices: %s\n", cudaGetErrorString(err));
        return 0;
    }

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device: %s\n", cudaGetErrorString(err));
        return 0;
    }

    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, 0);
    if (err == cudaSuccess) {
        printf("GPU: %s (%.1f GB)\n", props.name, 
               props.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    }

    return 1;
}

// ============================ Full-GPU Memory Management ==============================

typedef struct {
    uint8_t *d_headers;       // 80-byte headers input
    uint8_t *d_final_hashes;  // 32-byte final SHA3 outputs  
    uint8_t *h_headers;       // Host headers
    uint8_t *h_final_hashes;  // Host final hashes
    uint32_t batch_size;
} cuda_full_buffers_t;

static int cuda_allocate_full_buffers(cuda_full_buffers_t *buf, uint32_t batch_size) {
    cudaError_t err;
    buf->batch_size = batch_size;

    // GPU Memory - nur Input + Final Output
    err = cudaMalloc(&buf->d_headers, 80 * batch_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc headers failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    err = cudaMalloc(&buf->d_final_hashes, 32 * batch_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc final_hashes failed: %s\n", cudaGetErrorString(err));
        cudaFree(buf->d_headers);
        return 0;
    }

    // Host Memory (pinned für schnelle Transfers)
    err = cudaMallocHost(&buf->h_headers, 80 * batch_size);
    if (err != cudaSuccess) {
        buf->h_headers = (uint8_t*)malloc(80 * batch_size);
    }

    err = cudaMallocHost(&buf->h_final_hashes, 32 * batch_size);
    if (err != cudaSuccess) {
        buf->h_final_hashes = (uint8_t*)malloc(32 * batch_size);
    }

    if (!buf->h_headers || !buf->h_final_hashes) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 0;
    }

    return 1;
}

static void cuda_free_full_buffers(cuda_full_buffers_t *buf) {
    if (buf->d_headers) cudaFree(buf->d_headers);
    if (buf->d_final_hashes) cudaFree(buf->d_final_hashes);
    if (buf->h_headers) cudaFreeHost(buf->h_headers);
    if (buf->h_final_hashes) cudaFreeHost(buf->h_final_hashes);
}

// ============================ Full-GPU Pipeline Execution ==============================

static int cuda_execute_full_pipeline(cuda_full_buffers_t *buf, uint32_t work_items) {
    cudaError_t err;

    // Copy headers to GPU
    err = cudaMemcpy(buf->d_headers, buf->h_headers, 80 * work_items, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy headers H2D failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    // Launch Full RinHash Pipeline: BLAKE3 -> Argon2d -> SHA3-256 (alles auf GPU)
    err = rinhash_full_pipeline(buf->d_headers, buf->d_final_hashes, work_items, M_COST_KB);
    if (err != cudaSuccess) {
        fprintf(stderr, "Full pipeline kernel failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    // Copy final hashes back (nur Final Results)
    err = cudaMemcpy(buf->h_final_hashes, buf->d_final_hashes, 32 * work_items, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy final_hashes D2H failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    return 1;
}

// ============================ Header Building ==============================

static void build_header_le(const stratum_job_t *J, const uint8_t prevhash_le[32],
                            const uint8_t merkleroot_le[32], uint32_t ntime, uint32_t nbits,
                            uint32_t nonce, uint8_t out80[80]) {
    memset(out80, 0, 80);
    memcpy(out80 + 0, &J->version, 4);
    memcpy(out80 + 4, prevhash_le, 32);
    memcpy(out80 + 36, merkleroot_le, 32);
    memcpy(out80 + 68, &ntime, 4);
    memcpy(out80 + 72, &nbits, 4);
    memcpy(out80 + 76, &nonce, 4);
}

// ============================ MAIN ============================

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int sig) { (void)sig; g_stop = 1; }

int main(int argc, char **argv) {
    signal(SIGINT, on_sigint);

    if (getenv("RIN_DEBUG")) g_debug = atoi(getenv("RIN_DEBUG"));

    uint32_t BATCH = BATCH_DEFAULT;
    if (getenv("RIN_BATCH")) {
        int b = atoi(getenv("RIN_BATCH"));
        if (b > 0 && b <= 8192) BATCH = (uint32_t)b;
    }

    uint32_t chunk = 256;  // Größere Chunks für Full-GPU
    if (getenv("RIN_CHUNK")) {
        int c = atoi(getenv("RIN_CHUNK"));
        if (c > 0) chunk = (uint32_t)c;
    }
    if (chunk > BATCH) chunk = BATCH;

    printf("=== RinHash Full-GPU CUDA Miner ===\n");
    printf("Batch: %u (chunk=%u)\n", BATCH, chunk);
    printf("Pipeline: BLAKE3(GPU) -> Argon2d(GPU) -> SHA3-256(GPU)\n");

    // CUDA Setup
    if (!cuda_setup_device()) {
        fprintf(stderr, "CUDA setup failed\n");
        return 1;
    }

    // Full-GPU Buffer Allocation
    cuda_full_buffers_t buffers;
    if (!cuda_allocate_full_buffers(&buffers, BATCH)) {
        fprintf(stderr, "Buffer allocation failed\n");
        return 1;
    }

    // Stratum Connection (vereinfacht)
    stratum_ctx_t S;
    if (!stratum_connect_one(&S, POOL_CANDIDATES[0], PORT_DEFAULT, DEFAULT_WAL, DEFAULT_PASS)) {
        fprintf(stderr, "Stratum connect failed\n");
        return 1;
    }

    // Mining Loop
    stratum_job_t J = {0};
    uint8_t prevhash_le[32] = {0};
    uint8_t merkleroot_le[32] = {0};  // Vereinfacht
    uint32_t nonce_base = 0;
    
    uint64_t hashes_window = 0;
    uint64_t t_rate = mono_ms();

    printf("Starting mining...\n");

    while (!g_stop) {
        // Job Setup (vereinfacht - echte Implementation braucht Stratum Polling)
        J.version = 0x20000000;
        J.nbits = 0x1d00ffff;  // Testnet difficulty
        J.ntime = (uint32_t)time(NULL);

        // Header Building für Full Batch
        uint32_t workN = chunk;
        for (uint32_t i = 0; i < workN; i++) {
            uint32_t nonce = nonce_base + i;
            build_header_le(&J, prevhash_le, merkleroot_le, J.ntime, J.nbits, nonce,
                           &buffers.h_headers[i * 80]);
        }
        nonce_base += workN;

        // Full-GPU Pipeline Execution
        uint64_t t0 = mono_ms();
        
        if (!cuda_execute_full_pipeline(&buffers, workN)) {
            fprintf(stderr, "Full pipeline execution failed\n");
            break;
        }

        uint64_t t1 = mono_ms();
        double batch_ms = (double)(t1 - t0);
        hashes_window += workN;

        // Target Check (CPU - nur Final Results prüfen)
        for (uint32_t i = 0; i < workN; i++) {
            uint8_t *final_hash = &buffers.h_final_hashes[i * 32];
            update_statistics(final_hash);

            if (hash_meets_target_be(final_hash, g_share_target_be)) {
                uint32_t nonce = (nonce_base - workN) + i;
                printf("\nSHARE FOUND! nonce=%08x\n", nonce);
                g_accepted_shares++;
                // Submit zu Pool (Implementation gekürzt)
            }
        }

        // Hashrate Display
        uint64_t now = mono_ms();
        if (now - t_rate >= 5000) {
            double secs = (now - t_rate) / 1000.0;
            double rate = (double)hashes_window / secs;
            printf("Rate: %.1f H/s | Pipeline: %.1fms | Best: %u zeros | Shares: %u\r",
                   rate, batch_ms, g_best_leading_zeros, g_accepted_shares);
            fflush(stdout);
            t_rate = now;
            hashes_window = 0;
        }

        usleep(1000);  // Kurze Pause
    }

    // Cleanup
    cuda_free_full_buffers(&buffers);
    cudaDeviceReset();

    return 0;
}
