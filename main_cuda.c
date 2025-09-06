// main_cuda.c — Complete RinHash CUDA Miner
// Full GPU Pipeline: BLAKE3(GPU) -> Argon2d(GPU) -> SHA3-256(GPU) -> target check -> submit

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

// CUDA Kernel Declaration
extern "C" {
    cudaError_t rinhash_full_pipeline(
        const uint8_t* d_headers,
        uint8_t* d_final_hashes,
        uint32_t num_items,
        uint32_t m_cost_kb
    );
    void rinhash_cuda_cleanup(void);
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

#define M_COST_KB 64
#define T_COST    2
#define LANES     1
#define BATCH_DEFAULT 512

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

static uint64_t g_total_hashes = 0;
static uint32_t g_best_leading_zeros = 0;
static uint32_t g_accepted_shares = 0;
static uint32_t g_rejected_shares = 0;

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

static double target_to_diff(const uint8_t target_be[32]) {
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *diff1 = BN_bin2bn(DIFF1_TARGET_BE, 32, NULL);
    BIGNUM *tgt   = BN_bin2bn(target_be, 32, NULL);
    BIGNUM *q     = BN_new();

    if (BN_is_zero(tgt)) { 
        BN_free(diff1); BN_free(tgt); BN_free(q); BN_CTX_free(ctx); 
        return 0.0; 
    }
    BN_div(q, NULL, diff1, tgt, ctx);

    int n = BN_num_bytes(q);
    unsigned char buf[64] = {0};
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
    BN_bn2binpad(q, buf, n);
#else
    BN_bn2bin(q, buf + (64 - n));
#endif
    int off = (n >= 8) ? (n - 8) : 0;
    uint64_t top = 0; 
    for (int i=0; i<8 && (off+i)<n; i++) top = (top<<8) | buf[off+i];
    int rem = n - 8; 
    double d = (double)top; 
    while (rem-- > 0) d *= 256.0;

    BN_free(diff1); BN_free(tgt); BN_free(q); BN_CTX_free(ctx);
    return d;
}

static int hash_meets_target_be(const uint8_t hash_be[32], const uint8_t target_be[32]) {
    for (int i = 0; i < 32; i++) {
        if (hash_be[i] < target_be[i]) return 1;
        if (hash_be[i] > target_be[i]) return 0;
    }
    return 1;
}

// ============================ Stratum helpers ==============================

static int next_quoted(const char **pp, const char *end, char *out, size_t cap) {
    const char *p = *pp, *q1 = NULL, *q2 = NULL;
    for (; p < end; p++) if (*p == '"') { q1 = p; break; }
    if (!q1) return 0;
    for (p = q1 + 1; p < end; p++) if (*p == '"') { q2 = p; break; }
    if (!q2) return 0;
    size_t L = (size_t)(q2 - (q1 + 1)); if (L >= cap) L = cap - 1;
    memcpy(out, q1 + 1, L); out[L] = 0;
    *pp = q2 + 1;
    return 1;
}

typedef struct {
    char     job_id[128];
    char     prevhash_hex[65];
    char     coinb1_hex[4096];
    char     coinb2_hex[4096];
    char     merkle_hex[16][65];
    int      merkle_count;
    uint32_t version;
    uint32_t nbits;
    uint32_t ntime;
    int      clean;
} stratum_job_t;

static int stratum_parse_notify(const char *line, stratum_job_t *J) {
    if (!strstr(line, "\"mining.notify\"")) return 0;

    memset(J, 0, sizeof *J);

    const char *pp = strstr(line, "\"params\"");
    if (!pp) return 0;
    const char *lb = strchr(pp, '[');
    const char *rb = lb ? strrchr(pp, ']') : NULL;
    if (!lb || !rb || rb <= lb) return 0;

    const char *p = lb + 1;

    if (!next_quoted(&p, rb, J->job_id, sizeof J->job_id)) return 0;
    if (!next_quoted(&p, rb, J->prevhash_hex, sizeof J->prevhash_hex)) return 0;
    if (!next_quoted(&p, rb, J->coinb1_hex, sizeof J->coinb1_hex)) return 0;
    if (!next_quoted(&p, rb, J->coinb2_hex, sizeof J->coinb2_hex)) return 0;

    J->merkle_count = 0;
    const char *m_lb = strchr(p, '['), *m_rb = NULL;
    if (m_lb) {
        int depth = 0; 
        const char *scan = m_lb;
        for (; scan < rb; scan++) {
            if (*scan == '[') depth++;
            else if (*scan == ']') {
                depth--;
                if (depth == 0) { m_rb = scan; break; }
            }
        }
    }
    if (m_lb && m_rb && m_rb > m_lb) {
        const char *mp = m_lb + 1;
        while (J->merkle_count < 16) {
            char tmp[65];
            if (!next_quoted(&mp, m_rb, tmp, sizeof tmp)) break;
            snprintf(J->merkle_hex[J->merkle_count], 65, "%s", tmp);
            J->merkle_count++;
        }
        p = m_rb + 1;
    }

    char vhex[16]={0}, nbhex[16]={0}, nth[16]={0};
    if (!next_quoted(&p, rb, vhex, sizeof vhex)) return 0;  
    sscanf(vhex, "%x", &J->version);
    if (!next_quoted(&p, rb, nbhex, sizeof nbhex)) return 0; 
    sscanf(nbhex, "%x", &J->nbits);
    if (!next_quoted(&p, rb, nth, sizeof nth)) return 0;  
    sscanf(nth, "%x", &J->ntime);

    J->clean = strstr(p, "true") != NULL;
    return 1;
}

static int parse_set_difficulty_line(const char *line, double *out_diff) {
    const char *p = strstr(line, "mining.set_difficulty");
    if (!p) return 0;
    p = strstr(p, "\"params\""); if (!p) return 0;
    p = strchr(p, '['); if (!p) return 0; p++;
    while (*p==' '||*p=='\t') p++;
    char *endp=NULL;
    double d = strtod(p, &endp);
    if (endp == p) return 0;
    *out_diff = d;
    return 1;
}

static int parse_set_target_line(const char *line, uint8_t out_be[32]) {
    const char *p = strstr(line, "mining.set_target");
    if (!p) return 0;
    const char *pp = strstr(p, "\"params\""); if (!pp) return 0;
    const char *lb = strchr(pp, '['); 
    const char *rb = lb ? strchr(lb, ']') : NULL;
    if (!lb || !rb || rb <= lb) return 0;
    char thex[130] = {0};
    const char *scan = lb + 1;
    if (!next_quoted(&scan, rb, thex, sizeof thex)) return 0;
    if (!hex2bin(thex, out_be, 32)) return 0;
    return 1;
}

// ============================ Stratum core ==============================

typedef struct {
    char host[256];
    int  port;
    int  sock;
    char wallet[256];
    char pass[512];
    char extranonce1[64];
    uint32_t extranonce2_size;
} stratum_ctx_t;

static int set_nonblock(int s, int on) {
    int flags = fcntl(s, F_GETFL, 0);
    if (flags < 0) return -1;
    if (on) flags |= O_NONBLOCK; else flags &= ~O_NONBLOCK;
    return fcntl(s, F_SETFL, flags);
}

static int sock_connect_verbose(const char *host, int port, int timeout_ms) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) { perror("socket"); return -1; }
    
    struct sockaddr_in sa; 
    memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET; 
    sa.sin_port = htons((uint16_t)port);
    
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) {
        struct hostent *he = gethostbyname(host);
        if (!he) { close(s); return -1; }
        memcpy(&sa.sin_addr, he->h_addr, 4);
    }
    
    if (set_nonblock(s, 1) != 0) { perror("fcntl"); close(s); return -1; }
    
    int rc = connect(s, (struct sockaddr*)&sa, sizeof sa);
    if (rc != 0 && errno != EINPROGRESS) { perror("connect"); close(s); return -1; }
    
    fd_set wfds; FD_ZERO(&wfds); FD_SET(s, &wfds);
    struct timeval tv = { .tv_sec = timeout_ms/1000, .tv_usec = (timeout_ms%1000)*1000 };
    rc = select(s+1, NULL, &wfds, NULL, &tv);
    if (rc <= 0) { 
        if (rc==0) fprintf(stderr, "connect timeout %s\n", host); 
        else perror("select"); 
        close(s); 
        return -1; 
    }
    
    int soerr=0; socklen_t slen=sizeof soerr;
    if (getsockopt(s, SOL_SOCKET, SO_ERROR, &soerr, &slen) < 0 || soerr != 0) {
        fprintf(stderr, "connect() error: %s\n", soerr ? strerror(soerr) : "getsockopt");
        close(s); 
        return -1;
    }
    
    return s;
}

static char inbuf[65536];
static size_t inlen = 0;

static int send_line_verbose(int s, const char *line) {
    size_t L = strlen(line), o = 0;
    while (o < L) {
        ssize_t n = send(s, line + o, L - o, 0);
        if (n <= 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) { 
                usleep(1000); 
                continue; 
            }
            fprintf(stderr, "send() failed: %s\n", strerror(errno));
            return 0;
        }
        o += (size_t)n;
    }
    return 1;
}

static int recv_into_buffer(int s, int timeout_ms) {
    fd_set rfds; struct timeval tv;
    FD_ZERO(&rfds); FD_SET(s, &rfds);
    tv.tv_sec = timeout_ms/1000; tv.tv_usec = (timeout_ms%1000)*1000;
    int sel = select(s+1, &rfds, NULL, NULL, &tv);
    if (sel <= 0) return 0;
    char tmp[8192];
    ssize_t n = recv(s, tmp, sizeof tmp, MSG_DONTWAIT);
    if (n <= 0) return 0;
    if (inlen + (size_t)n >= sizeof inbuf) inlen = 0;
    memcpy(inbuf + inlen, tmp, (size_t)n); 
    inlen += (size_t)n;
    return 1;
}

static int next_line(char *out, size_t cap) {
    for (size_t i=0; i<inlen; i++) {
        if (inbuf[i]=='\n') {
            size_t L = i+1; if (L >= cap) L = cap-1;
            memcpy(out, inbuf, L); out[L]=0;
            memmove(inbuf, inbuf+i+1, inlen-(i+1));
            inlen -= (i+1);
            return 1;
        }
    }
    return 0;
}

static int parse_subscribe_result(const char *line, char *ex1, size_t ex1cap, uint32_t *ex2sz) {
    if (!strstr(line, "\"id\":1") || !strstr(line, "\"result\"")) return 0;
    const char *p = strstr(line, "\"result\""); if (!p) return 0;
    p = strchr(p, '['); if (!p) return 0; p++;
    if (*p != '[') return 0;
    int depth = 0;
    while (*p) {
        if (*p == '[') depth++;
        else if (*p == ']') { 
            depth--; 
            if (depth == 0) { p++; break; } 
        }
        p++;
    }
    if (depth != 0) return 0;
    while (*p == ' ' || *p == '\t' || *p == ',') p++;
    if (*p != '"') return 0;
    const char *q1 = ++p;
    while (*p && *p != '"') p++;
    if (*p != '"') return 0;
    size_t L = (size_t)(p - q1); if (L >= ex1cap) L = ex1cap - 1;
    memcpy(ex1, q1, L); ex1[L] = 0;
    p++;
    while (*p == ' ' || *p == '\t' || *p == ',') p++;
    if (!(*p == '-' || (*p >= '0' && *p <= '9'))) return 0;
    unsigned long v = strtoul(p, NULL, 10);
    if (v == 0 || v > 32) v = 4;
    *ex2sz = (uint32_t)v;
    return 1;
}

static int stratum_connect_one(stratum_ctx_t *C, const char *host, int port, 
                              const char *user, const char *pass) {
    memset(C, 0, sizeof *C);
    snprintf(C->host, sizeof C->host, "%s", host); 
    C->port = port;
    snprintf(C->wallet, sizeof C->wallet, "%s", user);
    snprintf(C->pass, sizeof C->pass, "%s", pass ? pass : "");

    C->sock = sock_connect_verbose(host, port, 5000);
    if (C->sock < 0) return 0;
    
    int one = 1; 
    setsockopt(C->sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    char sub[256];
    snprintf(sub, sizeof sub, "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"cpuminer-opt/3.21.0\"]}\n");
    if (!send_line_verbose(C->sock, sub)) { 
        fprintf(stderr, "Stratum send subscribe failed\n"); 
        close(C->sock); 
        return 0; 
    }

    time_t t0 = time(NULL); 
    char line[16384]; 
    int have_ex = 0;
    while (time(NULL) - t0 < 5) {
        recv_into_buffer(C->sock, 500);
        while (next_line(line, sizeof line)) {
            if (!have_ex && strstr(line, "\"result\"") && !strstr(line, "\"method\"")) {
                if (parse_subscribe_result(line, C->extranonce1, sizeof C->extranonce1, &C->extranonce2_size)) {
                    have_ex = 1;
                    continue;
                }
            }
            uint8_t t_be[32];
            if (parse_set_target_line(line, t_be)) {
                memcpy(g_share_target_be, t_be, 32);
                if (!g_diff_locked) g_share_diff = target_to_diff(g_share_target_be);
                if (g_debug) printf("[TARGET] set_target; approx diff=%.8f\n", g_share_diff);
                continue;
            }
            double dtmp;
            if (!g_diff_locked && parse_set_difficulty_line(line, &dtmp)) {
                g_share_diff = dtmp;
                diff_to_target(g_share_diff, g_share_target_be);
                if (g_debug) printf("[DIFF] set_difficulty=%.8f\n", g_share_diff);
                continue;
            }
        }
        if (have_ex) break;
    }
    if (!have_ex) { 
        fprintf(stderr, "No subscribe result within timeout (5s)\n"); 
        close(C->sock); 
        return 0; 
    }

    if (!C->extranonce1[0]) snprintf(C->extranonce1, sizeof C->extranonce1, "00000000");
    if (!C->extranonce2_size) C->extranonce2_size = 4;

    char auth[1536];
    int nw = snprintf(auth, sizeof auth, "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"%s\"]}\n", C->wallet, C->pass);
    if (nw <= 0 || (size_t)nw >= sizeof auth) { 
        fprintf(stderr, "authorize snprintf overflow\n"); 
        close(C->sock); 
        return 0; 
    }
    if (!send_line_verbose(C->sock, auth)) { 
        fprintf(stderr, "Stratum send authorize failed\n"); 
        close(C->sock); 
        return 0; 
    }

    time_t t1 = time(NULL);
    while (time(NULL) - t1 < 2) { 
        recv_into_buffer(C->sock, 200); 
        char dump[16384]; 
        while (next_line(dump, sizeof dump)) {} 
    }

    printf("Connected. extranonce1=%s ex2_size=%u\n", C->extranonce1, (unsigned)C->extranonce2_size);
    return 1;
}

static int stratum_connect_any(stratum_ctx_t *C, const char **hosts, int port, 
                              const char *user, const char *pass) {
    const char *env_host = getenv("POOL_HOST");
    const char *env_port = getenv("POOL_PORT");
    if (env_host && env_host[0]) {
        int p = (env_port && env_port[0]) ? atoi(env_port) : port;
        if (stratum_connect_one(C, env_host, p, user, pass)) return 1;
        fprintf(stderr, "ENV host failed, falling back to default list...\n");
    }
    for (int i = 0; hosts[i]; i++) {
        if (stratum_connect_one(C, hosts[i], port, user, pass)) return 1;
    }
    return 0;
}

static size_t build_submit_json(char *req, size_t cap, const char *wallet, const char *job_id,
                               const char *ex2_hex, uint32_t ntime_le, uint32_t nonce_le) {
    return (size_t)snprintf(req, cap,
        "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%s\",\"%08x\",\"%08x\"]}\n",
        wallet, job_id, ex2_hex, (unsigned)ntime_le, (unsigned)nonce_le
    );
}

static void stratum_wait_submit_ack(int sock) {
    char line[4096];
    uint64_t start = mono_ms();
    while (mono_ms() - start <= 5000) {
        recv_into_buffer(sock, 250);
        while (next_line(line, sizeof line)) {
            if (strstr(line, "\"id\":4")) {
                if (strstr(line, "\"result\":true")) { 
                    g_accepted_shares++; 
                    fprintf(stderr, " -> ACCEPTED\n"); 
                    fflush(stdout); 
                    return; 
                }
                if (strstr(line, "\"error\"")) { 
                    g_rejected_shares++; 
                    fprintf(stderr, "[REJECT] %s", line); 
                    fprintf(stderr, " -> REJECTED\n"); 
                    fflush(stdout); 
                    return; 
                }
            }
            double dtmp;
            if (!g_diff_locked && parse_set_difficulty_line(line, &dtmp)) {
                g_share_diff = dtmp; 
                diff_to_target(g_share_diff, g_share_target_be);
                if (g_debug) fprintf(stderr, "[DIFF] set_difficulty=%.8f\n", g_share_diff);
            } else {
                uint8_t t_be[32];
                if (!g_diff_locked && parse_set_target_line(line, t_be)) {
                    memcpy(g_share_target_be, t_be, 32);
                    g_share_diff = target_to_diff(g_share_target_be);
                    if (g_debug) fprintf(stderr, "[TARGET] set_target; approx diff=%.8f\n", g_share_diff);
                }
            }
        }
        usleep(20000);
    }
    fprintf(stderr, "[WARN] submit ack timeout (>5s)\n");
}

static int stratum_submit(stratum_ctx_t *C, const stratum_job_t *J,
                         const char *extranonce2_hex, uint32_t ntime_le, uint32_t nonce_le) {
    if (g_debug) {
        fprintf(stderr, "SUBMIT job=%s ex2=%s ntime=%08x nonce=%08x\n",
                J->job_id, extranonce2_hex, (unsigned)ntime_le, (unsigned)nonce_le);
    }
    char req[1536];
    size_t n = build_submit_json(req, sizeof req, C->wallet, J->job_id,
                                extranonce2_hex, ntime_le, nonce_le);
    if (n == 0 || n >= sizeof req) { 
        fprintf(stderr, "submit snprintf overflow\n"); 
        return 0; 
    }
    int ok = send_line_verbose(C->sock, req);
    if (ok) stratum_wait_submit_ack(C->sock);
    return ok;
}

// ============================ Merkle / Header Building ==============================

static void double_sha256(const uint8_t *in, size_t len, uint8_t out[32]) {
    uint8_t tmp[32];
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    
    EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);
    EVP_DigestUpdate(ctx, in, len);
    unsigned int olen = 0;
    EVP_DigestFinal_ex(ctx, tmp, &olen);
    
    EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);
    EVP_DigestUpdate(ctx, tmp, 32);
    olen = 0;
    EVP_DigestFinal_ex(ctx, out, &olen);
    
    EVP_MD_CTX_free(ctx);
}

static void build_merkle_root_le(const uint8_t cb_be[32], char merkle_hex[][65], int mcount, uint8_t out_le[32]) {
    uint8_t h_le[32]; 
    for (int i=0; i<32; i++) h_le[i] = cb_be[31-i];
    
    for (int i=0; i<mcount; i++) {
        uint8_t br_be[32], br_le[32], cat[64], dh[32];
        if (!hex2bin(merkle_hex[i], br_be, 32)) memset(br_be, 0, 32);
        for (int k=0; k<32; k++) br_le[k] = br_be[31-k];
        memcpy(cat, h_le, 32); 
        memcpy(cat+32, br_le, 32);
        double_sha256(cat, 64, dh);
        for (int k=0; k<32; k++) h_le[k] = dh[31-k];
    }
    memcpy(out_le, h_le, 32);
}

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
        printf("Compute Capability: %d.%d\n", props.major, props.minor);
    }

    return 1;
}

// ============================ CUDA Memory Management ==============================

typedef struct {
    uint8_t *d_headers;
    uint8_t *d_final_hashes;
    uint8_t *h_headers;
    uint8_t *h_final_hashes;
    uint32_t batch_size;
} cuda_full_buffers_t;

static int cuda_allocate_full_buffers(cuda_full_buffers_t *buf, uint32_t batch_size) {
    cudaError_t err;
    buf->batch_size = batch_size;

    // GPU Memory
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

static int cuda_execute_full_pipeline(cuda_full_buffers_t *buf, uint32_t work_items) {
    cudaError_t err;

    // Copy headers to GPU
    err = cudaMemcpy(buf->d_headers, buf->h_headers, 80 * work_items, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy headers H2D failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    // Launch Full RinHash Pipeline
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

    // Copy final hashes back
    err = cudaMemcpy(buf->h_final_hashes, buf->d_final_hashes, 32 * work_items, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy final_hashes D2H failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    return 1;
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

    uint32_t chunk = 256;
    if (getenv("RIN_CHUNK")) {
        int c = atoi(getenv("RIN_CHUNK"));
        if (c > 0) chunk = (uint32_t)c;
    }
    if (chunk > BATCH) chunk = BATCH;
    if (chunk < 1) chunk = 1;

    const char *wallet_env = getenv("WALLET");
    const char *pass_env = getenv("POOL_PASS");
    const char *WAL = wallet_env && wallet_env[0] ? wallet_env : DEFAULT_WAL;
    const char *PASS = pass_env && pass_env[0] ? pass_env : DEFAULT_PASS;

    printf("=== RinHash Full-GPU CUDA Miner ===\n");
    printf("Batch: %u (chunk=%u)\n", BATCH, chunk);
    printf("Wallet: %s\n", WAL);
    printf("Pipeline: BLAKE3(GPU) -> Argon2d(GPU) -> SHA3-256(GPU)\n");

    // CUDA Setup
    if (!cuda_setup_device()) {
        fprintf(stderr, "CUDA setup failed\n");
        return 1;
    }

    // Buffer Allocation
    cuda_full_buffers_t buffers;
    if (!cuda_allocate_full_buffers(&buffers, BATCH)) {
        fprintf(stderr, "Buffer allocation failed\n");
        return 1;
    }

    // Stratum Connection
    int PORT = getenv("POOL_PORT") ? atoi(getenv("POOL_PORT")) : PORT_DEFAULT;
    stratum_ctx_t S;
    if (!stratum_connect_any(&S, POOL_CANDIDATES, PORT, WAL, PASS)) {
        fprintf(stderr, "Stratum connect failed\n");
        return 1;
    }

    // Mining Variables
    uint8_t prevhash_le[32];
    stratum_job_t J = {0}, Jnew = {0};
    int have_job = 0;

    uint64_t hashes_window = 0;
    uint64_t t_rate = mono_ms();
    uint64_t t_poll = mono_ms();

    static uint32_t extranonce2_counter = 1;
    uint32_t en2_mask = (S.extranonce2_size >= 4) ? 0xFFFFFFFFu : ((1u << (S.extranonce2_size * 8)) - 1u);

    static uint32_t nonce_base = 0;
    int debug_shown_for_job = 0;

    while (!g_stop) {
        // Poll stratum
        if (mono_ms() - t_poll >= 50) {
            char line[16384];
            recv_into_buffer(S.sock, 0);
            while (next_line(line, sizeof line)) {
                double dtmp;
                if (!g_diff_locked && parse_set_difficulty_line(line, &dtmp)) {
                    g_share_diff = dtmp;
                    diff_to_target(g_share_diff, g_share_target_be);
                    if (g_debug) printf("[DIFF] set_difficulty=%.8f\n", g_share_diff);
                    continue;
                }
                if (strstr(line, "\"mining.notify\"")) {
                    if (stratum_parse_notify(line, &Jnew)) {
                        J = Jnew;
                        have_job = 1;
                        uint8_t prev_be[32];
                        hex2bin(J.prevhash_hex, prev_be, 32);
                        for (int i = 0; i < 32; i++) {
                            prevhash_le[i] = prev_be[31 - i];
                        }
                        nonce_base = 0;
                        debug_shown_for_job = 0;

                        g_job_gen++;
                        g_last_notify_clean = J.clean ? 1 : 0;
                        g_batch_counter = 0;

                        printf("New Job id %s%s\n", J.job_id, J.clean ? " (clean)" : "");
                    }
                }
            }
            t_poll = mono_ms();
        }
        
        if (!have_job) {
            usleep(10000);
            continue;
        }

        // Job snapshot for clean detection
        uint32_t batch_gen = g_job_gen;
        uint32_t clean_at_start = g_last_notify_clean;
        stratum_job_t batch_job = J;
        uint8_t batch_prevhash_le[32];
        memcpy(batch_prevhash_le, prevhash_le, 32);

        // Coinbase / Merkle
        uint8_t coinb1[4096], coinb2[4096];
        size_t cb1 = strlen(batch_job.coinb1_hex) / 2;
        size_t cb2 = strlen(batch_job.coinb2_hex) / 2;
        hex2bin(batch_job.coinb1_hex, coinb1, cb1);
        hex2bin(batch_job.coinb2_hex, coinb2, cb2);

        uint8_t en1[64]; 
        size_t en1b = strlen(S.extranonce1) / 2;
        if (en1b > 64) en1b = 64;
        hex2bin(S.extranonce1, en1, en1b);

        // Fresh extranonce2 per chunk
        uint32_t en2_val = (extranonce2_counter++) & en2_mask;

        uint8_t en2[64] = {0};
        for (int i = 0; i < (int)S.extranonce2_size && i < 4; i++)
            en2[i] = (uint8_t)((en2_val >> (8 * i)) & 0xFF);

        char en2_hex[64] = {0};
        for (uint32_t i = 0; i < S.extranonce2_size; i++)
            snprintf(en2_hex + i*2, sizeof en2_hex - i*2, "%02x", en2[i]);

        // Coinbase = coinb1 | en1 | en2 | coinb2
        uint8_t coinbase[8192]; 
        size_t off = 0;
        memcpy(coinbase + off, coinb1, cb1); off += cb1;
        memcpy(coinbase + off, en1, en1b); off += en1b;
        memcpy(coinbase + off, en2, S.extranonce2_size); off += S.extranonce2_size;
        memcpy(coinbase + off, coinb2, cb2); off += cb2;

        // Double-SHA256(coinbase) -> BE
        uint8_t cbh_be[32]; 
        double_sha256(coinbase, off, cbh_be);

        // Merkle-Root (LE) für Header
        uint8_t merkleroot_le[32];
        build_merkle_root_le(cbh_be, batch_job.merkle_hex, batch_job.merkle_count, merkleroot_le);

        // Time rolling
        static int nroll = -1;
        if (nroll < 0) {
            const char *env = getenv("RIN_NROLL");
            int m = env ? atoi(env) : 0;
            if (m < 0) m = 0; 
            if (m > 15) m = 15;
            nroll = m;
        }
        uint32_t submit_ntime = batch_job.ntime + (nroll ? (g_batch_counter % (uint32_t)nroll) : 0);

        // Header Building für GPU
        uint32_t workN = chunk;
        for (uint32_t i = 0; i < workN; i++) {
            uint32_t nonce = nonce_base + i;
            uint8_t header[80];
            build_header_le(&batch_job, batch_prevhash_le, merkleroot_le,
                           submit_ntime, batch_job.nbits, nonce, header);

            // Header direkt in GPU-Buffer
            memcpy(&buffers.h_headers[i * 80], header, 80);
        }
        nonce_base += workN;

        // GPU Pipeline Execution
        uint64_t t0 = mono_ms();
        
        if (!cuda_execute_full_pipeline(&buffers, workN)) {
            fprintf(stderr, "Full pipeline execution failed\n");
            break;
        }

        uint64_t t1 = mono_ms();
        double batch_ms = (double)(t1 - t0);
        hashes_window += workN;

        // Clean job check
        if (!clean_at_start && g_last_notify_clean && g_job_gen != batch_gen) {
            goto after_submit;
        }

        // Target Check
        for (uint32_t i = 0; i < workN; i++) {
            uint8_t *final_hash = &buffers.h_final_hashes[i * 32];
            update_statistics(final_hash);

            if (hash_meets_target_be(final_hash, g_share_target_be)) {
                uint32_t nonce = (nonce_base - workN) + i;
                if (g_debug || g_accepted_shares + g_rejected_shares < 5) {
                    printf("\nSHARE FOUND! job=%s nonce=%08x", J.job_id, nonce);
                    fflush(stdout);
                }
                stratum_submit(&S, &batch_job, en2_hex, submit_ntime, nonce);
            }
        }

after_submit:
        g_batch_counter++;

        if (g_debug && !debug_shown_for_job) {
            printf("Debug: job=%s best=%u zeros diff=%.6f batch=%u\n",
                   J.job_id, g_best_leading_zeros, g_share_diff, g_batch_counter);
            debug_shown_for_job = 1;
        }

        // Hashrate display
        uint64_t now = mono_ms();
        if (now - t_rate >= 5000) {
            double secs = (now - t_rate) / 1000.0;
            double rate = (double)hashes_window / secs;
            printf("Rate: %.1f H/s | Pipeline: %.1fms | Job: %s | Shares: %u/%u\r",
                   rate, batch_ms, J.job_id[0] ? J.job_id : "-", g_accepted_shares, g_rejected_shares);
            fflush(stdout);
            t_rate = now;
            hashes_window = 0;
        }
    }

    // Cleanup
    cuda_free_full_buffers(&buffers);
    rinhash_cuda_cleanup();
    cudaDeviceReset();
    close(S.sock);

    return 0;
}
