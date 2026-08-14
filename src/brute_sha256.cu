#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>

__constant__ char d_alphabet[64];
__constant__ uint32_t d_K[64];

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

__device__ __forceinline__ void sha256_compress(uint32_t* h, const unsigned char* chunk) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        int j = i * 4;
        w[i] = ((uint32_t)chunk[j] << 24) | ((uint32_t)chunk[j + 1] << 16) | ((uint32_t)chunk[j + 2] << 8) | ((uint32_t)chunk[j + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0];
    uint32_t b = h[1];
    uint32_t c = h[2];
    uint32_t d = h[3];
    uint32_t e = h[4];
    uint32_t f = h[5];
    uint32_t g = h[6];
    uint32_t hh = h[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = hh + S1 + ch + d_K[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        hh = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

// fast single-block path: valid for len <= 55
__device__ void sha256_device(const char* msg, int len, uint32_t* out) {
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    unsigned char chunk[64];
    for (int i = 0; i < len; ++i) chunk[i] = (unsigned char)msg[i];
    chunk[len] = 0x80;
    for (int i = len + 1; i < 56; ++i) chunk[i] = 0;
    uint64_t bitlen = (uint64_t)len * 8ULL;
    for (int i = 0; i < 8; ++i) chunk[63 - i] = (unsigned char)((bitlen >> (8 * i)) & 0xFF);
    sha256_compress(h, chunk);
    out[0] = h[0];
    out[1] = h[1];
    out[2] = h[2];
    out[3] = h[3];
    out[4] = h[4];
    out[5] = h[5];
    out[6] = h[6];
    out[7] = h[7];
}

// generic multi-block path: any length
__device__ void sha256_multi(const char* msg, int len, uint32_t* out) {
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    int total_blocks = (len + 72) / 64;
    for (int b = 0; b < total_blocks; ++b) {
        unsigned char chunk[64];
        int base = b * 64;
        for (int i = 0; i < 64; ++i) {
            int pos = base + i;
            if (pos < len) chunk[i] = (unsigned char)msg[pos];
            else if (pos == len) chunk[i] = 0x80;
            else chunk[i] = 0;
        }
        if (b == total_blocks - 1) {
            uint64_t bitlen = (uint64_t)len * 8ULL;
            for (int i = 0; i < 8; ++i) chunk[63 - i] = (unsigned char)((bitlen >> (8 * i)) & 0xFF);
        }
        sha256_compress(h, chunk);
    }
    out[0] = h[0];
    out[1] = h[1];
    out[2] = h[2];
    out[3] = h[3];
    out[4] = h[4];
    out[5] = h[5];
    out[6] = h[6];
    out[7] = h[7];
}

__global__ void stress_kernel(int L, int N, unsigned long long goal, unsigned long long* ops_counter) {
    unsigned long long start = clock64();
    char buf[32];
    uint32_t acc = 0u;
    unsigned long long s = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    while (clock64() - start < goal) {
        s += 1ULL;
        for (int i = 0; i < L; ++i) buf[i] = d_alphabet[(s + (unsigned long long)i) % (unsigned long long)N];
        uint32_t h[8];
        sha256_device(buf, L, h);
        acc += h[0];                 // sink: keep the hash alive (was DCE'd before)
        atomicAdd(ops_counter, 1ULL);
    }
    if (acc == 0xDEADBEEFu) atomicAdd(ops_counter, 1ULL);   // never true, but compiler can't prove it
}

// compile-time-L carry-increment enumeration (generalizes brute_kernel7 to any L):
// each thread walks a contiguous chunk of indices, maintaining the base-N digits
// incrementally (add+compare+subtract, no div/mod in the hot loop).
template<int L>
__global__ void brute_kernel_t(int N, uint64_t start, uint64_t limit, const uint32_t* target, long long* found_idx, char* found_pwd, unsigned long long* ops_counter) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nthreads = (uint64_t)blockDim.x * gridDim.x;
    uint64_t chunk = (limit + nthreads - 1ULL) / nthreads;
    uint64_t lo = start + tid * chunk;
    uint64_t hi = lo + chunk;
    uint64_t end = start + limit;
    if (hi > end) hi = end;
    if (lo >= hi) return;
    unsigned int d[L];
    uint64_t t = lo;
#pragma unroll
    for (int i = 0; i < L; ++i) { d[i] = (unsigned int)(t % (uint64_t)N); t /= (uint64_t)N; }
    char buf[L];
    uint64_t idx = lo;
    while (idx < hi) {
        if (*found_idx != -1) return;
#pragma unroll
        for (int i = 0; i < L; ++i) buf[i] = d_alphabet[d[i]];
        uint32_t h[8];
        sha256_device(buf, L, h);      // L is a template constant: folds to the fast fixed-length path
        atomicAdd(ops_counter, 1ULL);
        bool eq = true;
        for (int i = 0; i < 8; ++i) { if (h[i] != target[i]) { eq = false; break; } }
        if (eq) {
            if (atomicCAS((unsigned long long*)found_idx, ~0ULL, (unsigned long long)idx) == ~0ULL) {
#pragma unroll
                for (int i = 0; i < L; ++i) found_pwd[i] = buf[i];
                found_pwd[L] = '\0';
            }
            return;
        }
        unsigned int carry = 1u;
#pragma unroll
        for (int i = 0; i < L; ++i) {
            unsigned int s = d[i] + carry;
            if (s >= (unsigned int)N) { s -= (unsigned int)N; carry = 1u; } else { carry = 0u; }
            d[i] = s;
        }
        idx += 1ULL;
    }
}

__global__ void dict_kernel(const char* flat, const int* offsets, const int* lens, int count, long long base, const uint32_t* target, long long* found_idx, char* found_pwd, unsigned long long* ops_counter) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    while (idx < (uint64_t)count) {
        if (*found_idx != -1) return;
        const char* msg = flat + offsets[(int)idx];
        int len = lens[(int)idx];
        uint32_t h[8];
        if (len <= 55) sha256_device(msg, len, h);
        else sha256_multi(msg, len, h);
        atomicAdd(ops_counter, 1ULL);
        bool eq = true;
        for (int i = 0; i < 8; ++i) { if (h[i] != target[i]) { eq = false; break; } }
        if (eq) {
            if (atomicCAS((unsigned long long*)found_idx, ~0ULL, (unsigned long long)(base + (long long)idx)) == ~0ULL) {
                for (int i = 0; i < len; ++i) found_pwd[i] = msg[i];
                found_pwd[len] = '\0';
            }
            return;
        }
        idx += stride;
    }
}

// fixed-length dict kernel: L is compile-time, single-block fast path only (L <= 55)
template<int L>
__global__ void dict_kernel_fixed(const char* flat, const int* offsets, int count, long long base, const uint32_t* target, long long* found_idx, char* found_pwd, unsigned long long* ops_counter) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    while (idx < (uint64_t)count) {
        if (*found_idx != -1) return;
        const char* msg = flat + offsets[(int)idx];
        char buf[L];
#pragma unroll
        for (int i = 0; i < L; ++i) buf[i] = msg[i];
        uint32_t h[8];
        sha256_device(buf, L, h);      // L literal -> fast fixed-length path
        atomicAdd(ops_counter, 1ULL);
        bool eq = true;
        for (int i = 0; i < 8; ++i) { if (h[i] != target[i]) { eq = false; break; } }
        if (eq) {
            if (atomicCAS((unsigned long long*)found_idx, ~0ULL, (unsigned long long)(base + (long long)idx)) == ~0ULL) {
#pragma unroll
                for (int i = 0; i < L; ++i) found_pwd[i] = buf[i];
                found_pwd[L] = '\0';
            }
            return;
        }
        idx += stride;
    }
}

void sha256_host(const char* msg, int len, uint32_t* out) {
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    auto rotr = [](uint32_t x, int n){ return (x >> n) | (x << (32 - n)); };
    uint32_t h0 = 0x6a09e667;
    uint32_t h1 = 0xbb67ae85;
    uint32_t h2 = 0x3c6ef372;
    uint32_t h3 = 0xa54ff53a;
    uint32_t h4 = 0x510e527f;
    uint32_t h5 = 0x9b05688c;
    uint32_t h6 = 0x1f83d9ab;
    uint32_t h7 = 0x5be0cd19;
    int total_blocks = (len + 72) / 64;
    for (int b = 0; b < total_blocks; ++b) {
        unsigned char chunk[64];
        int base = b * 64;
        for (int i = 0; i < 64; ++i) {
            int pos = base + i;
            if (pos < len) chunk[i] = (unsigned char)msg[pos];
            else if (pos == len) chunk[i] = 0x80;
            else chunk[i] = 0;
        }
        if (b == total_blocks - 1) {
            uint64_t bitlen = (uint64_t)len * 8ULL;
            for (int i = 0; i < 8; ++i) chunk[63 - i] = (unsigned char)((bitlen >> (8 * i)) & 0xFF);
        }
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            int j = i * 4;
            w[i] = ((uint32_t)chunk[j] << 24) | ((uint32_t)chunk[j + 1] << 16) | ((uint32_t)chunk[j + 2] << 8) | ((uint32_t)chunk[j + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h0;
        uint32_t b2 = h1;
        uint32_t c = h2;
        uint32_t d = h3;
        uint32_t e = h4;
        uint32_t f = h5;
        uint32_t g = h6;
        uint32_t h = h7;
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b2) ^ (a & c) ^ (b2 & c);
            uint32_t temp2 = S0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b2;
            b2 = a;
            a = temp1 + temp2;
        }
        h0 += a;
        h1 += b2;
        h2 += c;
        h3 += d;
        h4 += e;
        h5 += f;
        h6 += g;
        h7 += h;
    }
    out[0] = h0;
    out[1] = h1;
    out[2] = h2;
    out[3] = h3;
    out[4] = h4;
    out[5] = h5;
    out[6] = h6;
    out[7] = h7;
}

template<int L>
void launch_brute_kernel(int N, uint64_t start, uint64_t limit, int blocks, int threads,
                         uint32_t* d_target, long long* d_found_idx, char* d_found_pwd, unsigned long long* d_ops) {
    brute_kernel_t<L><<<blocks, threads>>>(N, start, limit, d_target, d_found_idx, d_found_pwd, d_ops);
}

void dispatch_brute(int L, int N, uint64_t start, uint64_t limit, int blocks, int threads,
                    uint32_t* d_target, long long* d_found_idx, char* d_found_pwd, unsigned long long* d_ops) {
    switch (L) {
        case 1:  launch_brute_kernel<1>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 2:  launch_brute_kernel<2>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 3:  launch_brute_kernel<3>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 4:  launch_brute_kernel<4>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 5:  launch_brute_kernel<5>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 6:  launch_brute_kernel<6>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 7:  launch_brute_kernel<7>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 8:  launch_brute_kernel<8>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 9:  launch_brute_kernel<9>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 10: launch_brute_kernel<10>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 11: launch_brute_kernel<11>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 12: launch_brute_kernel<12>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 13: launch_brute_kernel<13>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 14: launch_brute_kernel<14>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 15: launch_brute_kernel<15>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 16: launch_brute_kernel<16>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 17: launch_brute_kernel<17>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 18: launch_brute_kernel<18>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 19: launch_brute_kernel<19>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 20: launch_brute_kernel<20>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 21: launch_brute_kernel<21>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 22: launch_brute_kernel<22>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 23: launch_brute_kernel<23>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 24: launch_brute_kernel<24>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 25: launch_brute_kernel<25>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 26: launch_brute_kernel<26>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 27: launch_brute_kernel<27>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 28: launch_brute_kernel<28>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 29: launch_brute_kernel<29>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 30: launch_brute_kernel<30>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 31: launch_brute_kernel<31>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        case 32: launch_brute_kernel<32>(N, start, limit, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops); break;
        default: std::cerr << "invalid L for dispatch: " << L << std::endl; break;
    }
}

uint64_t pow_uint64(uint64_t base, int exp) { uint64_t r = 1; for (int i = 0; i < exp; ++i) r *= base; return r; }

void idx_to_string(uint64_t idx, int L, const std::string& alphabet, std::string& out) {
    out.resize(L);
    uint64_t N = alphabet.size();
    for (int i = 0; i < L; ++i) { out[i] = alphabet[idx % N]; idx /= N; }
}

static const uint32_t K64[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--hash") {
        std::string s = argc >= 3 ? std::string(argv[2]) : std::string("XXZXNFZ");
        uint32_t hv[8];
        sha256_host(s.c_str(), (int)s.size(), hv);
        std::ostringstream oss;
        oss << std::hex << std::nouppercase << std::setfill('0');
        for (int i = 0; i < 8; ++i) oss << std::setw(8) << hv[i];
        std::cout << "SHA256(" << s << ")=" << oss.str() << std::endl;
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--stress") {
        int duration = argc >= 3 ? std::stoi(argv[2]) : 10;
        int sL = argc >= 4 ? std::stoi(argv[3]) : 7;
        std::string sAlpha = argc >= 5 ? std::string(argv[4]) : std::string("abcdefghijklmnopqrstuvwxyz");
        if (sL > 32 || sAlpha.size() > 64) { std::cerr << "invalid params" << std::endl; return 1; }
        char h_alphabet2[64];
        for (size_t i = 0; i < sAlpha.size(); ++i) h_alphabet2[i] = sAlpha[i];
        cudaMemcpyToSymbol(d_alphabet, h_alphabet2, 64);
        cudaMemcpyToSymbol(d_K, K64, sizeof(K64));
        int clock_khz = 0;
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
        unsigned long long goal = (unsigned long long)clock_khz * 1000ULL * (unsigned long long)duration;
        unsigned long long* d_ops2; cudaMalloc(&d_ops2, sizeof(unsigned long long));
        unsigned long long zero2 = 0ULL; cudaMemcpy(d_ops2, &zero2, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        int threads2 = 256;
        int sm_count = 0;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
        int max_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, stress_kernel, threads2, 0);
        int blocks2 = max_blocks_per_sm * sm_count;
        if (blocks2 < 1) blocks2 = 1;
        std::cout << "STRESS blocks=" << blocks2 << " threads=" << threads2 << " sm_count=" << sm_count << " max_blocks_per_sm=" << max_blocks_per_sm << " goal_cycles=" << goal << std::endl;
        auto hs = std::chrono::high_resolution_clock::now();
        stress_kernel<<<blocks2, threads2>>>(sL, (int)sAlpha.size(), goal, d_ops2);
        cudaDeviceSynchronize();
        auto he = std::chrono::high_resolution_clock::now();
        double sec2 = std::chrono::duration<double>(he - hs).count();
        unsigned long long hops2=0; cudaMemcpy(&hops2, d_ops2, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        std::cout << "STRESS length=" << sL << " alphabet=" << sAlpha.size() << " secs=" << duration << std::endl;
        std::cout << "GPU ops=" << hops2 << " time(s)=" << sec2 << " hashes/s=" << (unsigned long long)(hops2/sec2) << std::endl;
        cudaFree(d_ops2);
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--dict") {
        std::string dictPath = argc >= 3 ? std::string(argv[2]) : std::string("candidates.txt");
        bool markov = true;
        std::vector<std::string> targets;
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--no-markov") markov = false;
            else targets.push_back(a);
        }
        if (targets.empty()) targets.push_back("password");
        cudaMemcpyToSymbol(d_K, K64, sizeof(K64));
        std::ifstream in(dictPath, std::ios::binary | std::ios::ate);
        if (!in) { std::cerr << "open dict failed" << std::endl; return 1; }
        std::streamoff fsize = in.tellg();
        in.seekg(0, std::ios::beg);
        const int MAX_PWD = 1024;
        const int MAX_SINGLE = 55;
        // P2: read the whole file directly into PINNED host memory (single pass, no
        // pageable intermediate) so the H2D upload runs at ~2x pageable bandwidth.
        char* h_flat = nullptr;
        cudaMallocHost(&h_flat, (size_t)fsize + 1);
        in.read(h_flat, fsize);
        in.close();
        // pass 1: count lines per length bucket + train 2nd-order Markov model
        // (digram counts with 1st-order backoff; prev=0 models line start)
        std::vector<int> bucket_count(MAX_SINGLE + 1, 0);
        long long long_count = 0, skipped = 0;
        std::vector<unsigned int> unigram(256, 0);
        std::vector<unsigned int> bigram(256 * 256, 0);
        {
            long long i = 0;
            while (i < fsize) {
                long long s = i;
                while (i < fsize && h_flat[i] != '\n') ++i;
                long long e = i;
                if (e > s && h_flat[e - 1] == '\r') e -= 1;   // strip CR of CRLF files
                int len = (int)(e - s);
                if (len == 0) { }
                else if (len > MAX_PWD) skipped++;
                else if (len <= MAX_SINGLE) bucket_count[len]++;
                else long_count++;
                if (markov && len >= 1 && len <= MAX_PWD) {
                    unsigned char prev = 0;
                    for (long long p = s; p < e; ++p) {
                        unsigned char c = (unsigned char)h_flat[p];
                        unigram[c]++;
                        bigram[prev * 256 + c]++;
                        prev = c;
                    }
                }
                if (i < fsize) ++i;
            }
        }
        int count = (int)long_count;
        for (int L = 1; L <= MAX_SINGLE; ++L) count += bucket_count[L];
        if (count == 0) { std::cerr << "no candidates" << std::endl; cudaFreeHost(h_flat); return 1; }
        std::cout << "DICT loaded lines=" << count << " bytes=" << fsize << " long(>55)=" << long_count
                  << " skipped(>" << MAX_PWD << ")=" << skipped << " markov=" << (markov ? "on" : "off") << std::endl;
        // pass 2: fill pinned offset/lens arrays in physical bucket layout (1..55 then long),
        // and record the original file line number of every entry (host side only).
        int* h_offsets = nullptr; cudaMallocHost(&h_offsets, (size_t)count * sizeof(int));
        int* h_lens = nullptr;    cudaMallocHost(&h_lens, (size_t)(long_count + 1) * sizeof(int));
        int* h_origline = new int[count];
        std::vector<int> phys_base(MAX_SINGLE + 1, 0);
        std::vector<int> fill_pos(MAX_SINGLE + 1, 0);
        int cur = 0;
        for (int L = 1; L <= MAX_SINGLE; ++L) { phys_base[L] = cur; fill_pos[L] = cur; cur += bucket_count[L]; }
        int long_base = cur;
        {
            long long long_pos = 0;
            long long i = 0;
            long long line_no = 0;
            while (i < fsize) {
                long long s = i;
                while (i < fsize && h_flat[i] != '\n') ++i;
                long long e = i;
                if (e > s && h_flat[e - 1] == '\r') e -= 1;
                int len = (int)(e - s);
                if (len >= 1 && len <= MAX_PWD) {
                    if (len <= MAX_SINGLE) {
                        int p = fill_pos[len]++;
                        h_offsets[p] = (int)s;
                        h_origline[p] = (int)line_no;
                    } else {
                        h_offsets[long_base + long_pos] = (int)s;
                        h_origline[long_base + long_pos] = (int)line_no;
                        h_lens[long_pos] = len;
                        long_pos++;
                    }
                }
                line_no++;
                if (i < fsize) ++i;
            }
        }
        // P4 (Markov ordering): score every entry with the trained model, sort each
        // bucket by score desc, and order bucket launches by descending best-score.
        // This approximates global probability order while keeping per-length kernels.
        auto score_line = [&](int off, int len) -> float {
            long long tot = 0;
            for (int i = 0; i < 256; ++i) tot += unigram[i];
            float sc = 0.0f;
            unsigned char prev = 0;
            for (int p = off; p < off + len; ++p) {
                unsigned char c = (unsigned char)h_flat[p];
                float prob = (unigram[prev] > 0)
                    ? (float)(bigram[prev * 256 + c] + 1) / (float)(unigram[prev] + 256)
                    : (float)(unigram[c] + 1) / (float)(tot + 256);
                sc += logf(prob);
                prev = c;
            }
            return sc;
        };
        std::vector<float> bucket_max(MAX_SINGLE + 1, -1.0e30f);
        std::vector<int> order;
        if (markov) {
            std::vector<std::pair<float, int>> scored;   // (score, physical position)
            std::vector<int> new_off, new_line;
            for (int L = 1; L <= MAX_SINGLE; ++L) {
                int n = bucket_count[L];
                if (n == 0) continue;
                scored.resize(n);
                for (int k = 0; k < n; ++k) {
                    int p = phys_base[L] + k;
                    scored[k] = std::make_pair(score_line(h_offsets[p], L), p);
                }
                std::sort(scored.begin(), scored.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
                new_off.resize(n);
                new_line.resize(n);
                for (int k = 0; k < n; ++k) { new_off[k] = h_offsets[scored[k].second]; new_line[k] = h_origline[scored[k].second]; }
                for (int k = 0; k < n; ++k) { h_offsets[phys_base[L] + k] = new_off[k]; h_origline[phys_base[L] + k] = new_line[k]; }
                bucket_max[L] = scored[0].first;
            }
        } else {
            for (int L = 1; L <= MAX_SINGLE; ++L) if (bucket_count[L] > 0) bucket_max[L] = 0.0f;
        }
        for (int L = 1; L <= MAX_SINGLE; ++L) if (bucket_count[L] > 0) order.push_back(L);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return bucket_max[a] > bucket_max[b]; });
        // attack-order bases: rank of the first entry of each bucket in the launch sequence
        std::vector<int> attack_base(MAX_SINGLE + 1, 0);
        int acc = 0;
        for (size_t k = 0; k < order.size(); ++k) { attack_base[order[k]] = acc; acc += bucket_count[order[k]]; }
        int long_attack_base = acc;
        // upload everything ONCE; it stays resident in VRAM for all targets
        uint32_t* d_target; cudaMalloc(&d_target, 8 * sizeof(uint32_t));
        long long* d_found_idx; cudaMalloc(&d_found_idx, sizeof(long long));
        char* d_found_pwd; cudaMalloc(&d_found_pwd, MAX_PWD + 1);
        unsigned long long* d_ops; cudaMalloc(&d_ops, sizeof(unsigned long long));
        char* d_flat; cudaMalloc(&d_flat, (size_t)fsize); cudaMemcpy(d_flat, h_flat, (size_t)fsize, cudaMemcpyHostToDevice);
        int* d_offsets; cudaMalloc(&d_offsets, (size_t)count * sizeof(int)); cudaMemcpy(d_offsets, h_offsets, (size_t)count * sizeof(int), cudaMemcpyHostToDevice);
        int* d_lens; cudaMalloc(&d_lens, (size_t)(long_count + 1) * sizeof(int)); cudaMemcpy(d_lens, h_lens, (size_t)(long_count + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        const int threads = 256;
        for (size_t ti = 0; ti < targets.size(); ++ti) {
            uint32_t target[8];
            sha256_host(targets[ti].c_str(), (int)targets[ti].size(), target);
            cudaMemcpy(d_target, target, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
            long long h_found_idx = -1;
            cudaMemcpy(d_found_idx, &h_found_idx, sizeof(long long), cudaMemcpyHostToDevice);
            unsigned long long zero = 0ULL;
            cudaMemcpy(d_ops, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);
            cudaEventRecord(start);
            // P3: one compile-time-L kernel per length bucket (fast path), then the
            // generic multi-block kernel for the rare >55-byte lines.
            for (size_t ok = 0; ok < order.size(); ++ok) {
                int L = order[ok];
                if (bucket_count[L] == 0) continue;
                int b = (bucket_count[L] + threads - 1) / threads; if (b > 65535) b = 65535; if (b < 1) b = 1;
                switch (L) {
                    case 1:  dict_kernel_fixed<1><<<b, threads>>>(d_flat, d_offsets + phys_base[1],  bucket_count[1],  attack_base[1],  d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 2:  dict_kernel_fixed<2><<<b, threads>>>(d_flat, d_offsets + phys_base[2],  bucket_count[2],  attack_base[2],  d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 3:  dict_kernel_fixed<3><<<b, threads>>>(d_flat, d_offsets + phys_base[3],  bucket_count[3],  attack_base[3],  d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 4:  dict_kernel_fixed<4><<<b, threads>>>(d_flat, d_offsets + phys_base[4],  bucket_count[4],  attack_base[4],  d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 5:  dict_kernel_fixed<5><<<b, threads>>>(d_flat, d_offsets + phys_base[5],  bucket_count[5],  attack_base[5],  d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 6:  dict_kernel_fixed<6><<<b, threads>>>(d_flat, d_offsets + phys_base[6],  bucket_count[6],  attack_base[6],  d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 7:  dict_kernel_fixed<7><<<b, threads>>>(d_flat, d_offsets + phys_base[7],  bucket_count[7],  attack_base[7],  d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 8:  dict_kernel_fixed<8><<<b, threads>>>(d_flat, d_offsets + phys_base[8],  bucket_count[8],  attack_base[8],  d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 9:  dict_kernel_fixed<9><<<b, threads>>>(d_flat, d_offsets + phys_base[9],  bucket_count[9],  attack_base[9],  d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 10: dict_kernel_fixed<10><<<b, threads>>>(d_flat, d_offsets + phys_base[10], bucket_count[10], attack_base[10], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 11: dict_kernel_fixed<11><<<b, threads>>>(d_flat, d_offsets + phys_base[11], bucket_count[11], attack_base[11], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 12: dict_kernel_fixed<12><<<b, threads>>>(d_flat, d_offsets + phys_base[12], bucket_count[12], attack_base[12], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 13: dict_kernel_fixed<13><<<b, threads>>>(d_flat, d_offsets + phys_base[13], bucket_count[13], attack_base[13], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 14: dict_kernel_fixed<14><<<b, threads>>>(d_flat, d_offsets + phys_base[14], bucket_count[14], attack_base[14], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 15: dict_kernel_fixed<15><<<b, threads>>>(d_flat, d_offsets + phys_base[15], bucket_count[15], attack_base[15], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 16: dict_kernel_fixed<16><<<b, threads>>>(d_flat, d_offsets + phys_base[16], bucket_count[16], attack_base[16], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 17: dict_kernel_fixed<17><<<b, threads>>>(d_flat, d_offsets + phys_base[17], bucket_count[17], attack_base[17], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 18: dict_kernel_fixed<18><<<b, threads>>>(d_flat, d_offsets + phys_base[18], bucket_count[18], attack_base[18], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 19: dict_kernel_fixed<19><<<b, threads>>>(d_flat, d_offsets + phys_base[19], bucket_count[19], attack_base[19], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 20: dict_kernel_fixed<20><<<b, threads>>>(d_flat, d_offsets + phys_base[20], bucket_count[20], attack_base[20], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 21: dict_kernel_fixed<21><<<b, threads>>>(d_flat, d_offsets + phys_base[21], bucket_count[21], attack_base[21], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 22: dict_kernel_fixed<22><<<b, threads>>>(d_flat, d_offsets + phys_base[22], bucket_count[22], attack_base[22], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 23: dict_kernel_fixed<23><<<b, threads>>>(d_flat, d_offsets + phys_base[23], bucket_count[23], attack_base[23], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 24: dict_kernel_fixed<24><<<b, threads>>>(d_flat, d_offsets + phys_base[24], bucket_count[24], attack_base[24], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 25: dict_kernel_fixed<25><<<b, threads>>>(d_flat, d_offsets + phys_base[25], bucket_count[25], attack_base[25], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 26: dict_kernel_fixed<26><<<b, threads>>>(d_flat, d_offsets + phys_base[26], bucket_count[26], attack_base[26], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 27: dict_kernel_fixed<27><<<b, threads>>>(d_flat, d_offsets + phys_base[27], bucket_count[27], attack_base[27], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 28: dict_kernel_fixed<28><<<b, threads>>>(d_flat, d_offsets + phys_base[28], bucket_count[28], attack_base[28], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 29: dict_kernel_fixed<29><<<b, threads>>>(d_flat, d_offsets + phys_base[29], bucket_count[29], attack_base[29], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 30: dict_kernel_fixed<30><<<b, threads>>>(d_flat, d_offsets + phys_base[30], bucket_count[30], attack_base[30], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 31: dict_kernel_fixed<31><<<b, threads>>>(d_flat, d_offsets + phys_base[31], bucket_count[31], attack_base[31], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 32: dict_kernel_fixed<32><<<b, threads>>>(d_flat, d_offsets + phys_base[32], bucket_count[32], attack_base[32], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 33: dict_kernel_fixed<33><<<b, threads>>>(d_flat, d_offsets + phys_base[33], bucket_count[33], attack_base[33], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 34: dict_kernel_fixed<34><<<b, threads>>>(d_flat, d_offsets + phys_base[34], bucket_count[34], attack_base[34], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 35: dict_kernel_fixed<35><<<b, threads>>>(d_flat, d_offsets + phys_base[35], bucket_count[35], attack_base[35], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 36: dict_kernel_fixed<36><<<b, threads>>>(d_flat, d_offsets + phys_base[36], bucket_count[36], attack_base[36], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 37: dict_kernel_fixed<37><<<b, threads>>>(d_flat, d_offsets + phys_base[37], bucket_count[37], attack_base[37], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 38: dict_kernel_fixed<38><<<b, threads>>>(d_flat, d_offsets + phys_base[38], bucket_count[38], attack_base[38], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 39: dict_kernel_fixed<39><<<b, threads>>>(d_flat, d_offsets + phys_base[39], bucket_count[39], attack_base[39], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 40: dict_kernel_fixed<40><<<b, threads>>>(d_flat, d_offsets + phys_base[40], bucket_count[40], attack_base[40], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 41: dict_kernel_fixed<41><<<b, threads>>>(d_flat, d_offsets + phys_base[41], bucket_count[41], attack_base[41], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 42: dict_kernel_fixed<42><<<b, threads>>>(d_flat, d_offsets + phys_base[42], bucket_count[42], attack_base[42], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 43: dict_kernel_fixed<43><<<b, threads>>>(d_flat, d_offsets + phys_base[43], bucket_count[43], attack_base[43], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 44: dict_kernel_fixed<44><<<b, threads>>>(d_flat, d_offsets + phys_base[44], bucket_count[44], attack_base[44], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 45: dict_kernel_fixed<45><<<b, threads>>>(d_flat, d_offsets + phys_base[45], bucket_count[45], attack_base[45], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 46: dict_kernel_fixed<46><<<b, threads>>>(d_flat, d_offsets + phys_base[46], bucket_count[46], attack_base[46], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 47: dict_kernel_fixed<47><<<b, threads>>>(d_flat, d_offsets + phys_base[47], bucket_count[47], attack_base[47], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 48: dict_kernel_fixed<48><<<b, threads>>>(d_flat, d_offsets + phys_base[48], bucket_count[48], attack_base[48], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 49: dict_kernel_fixed<49><<<b, threads>>>(d_flat, d_offsets + phys_base[49], bucket_count[49], attack_base[49], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 50: dict_kernel_fixed<50><<<b, threads>>>(d_flat, d_offsets + phys_base[50], bucket_count[50], attack_base[50], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 51: dict_kernel_fixed<51><<<b, threads>>>(d_flat, d_offsets + phys_base[51], bucket_count[51], attack_base[51], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 52: dict_kernel_fixed<52><<<b, threads>>>(d_flat, d_offsets + phys_base[52], bucket_count[52], attack_base[52], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 53: dict_kernel_fixed<53><<<b, threads>>>(d_flat, d_offsets + phys_base[53], bucket_count[53], attack_base[53], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 54: dict_kernel_fixed<54><<<b, threads>>>(d_flat, d_offsets + phys_base[54], bucket_count[54], attack_base[54], d_target, d_found_idx, d_found_pwd, d_ops); break;
                    case 55: dict_kernel_fixed<55><<<b, threads>>>(d_flat, d_offsets + phys_base[55], bucket_count[55], attack_base[55], d_target, d_found_idx, d_found_pwd, d_ops); break;
                }
            }
            if (long_count > 0) {
                int b = (int)((long_count + threads - 1) / threads); if (b > 65535) b = 65535; if (b < 1) b = 1;
                dict_kernel<<<b, threads>>>(d_flat, d_offsets + long_base, d_lens, (int)long_count, (long long)long_attack_base, d_target, d_found_idx, d_found_pwd, d_ops);
            }
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop);
            unsigned long long hops = 0ULL; cudaMemcpy(&hops, d_ops, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_found_idx, d_found_idx, sizeof(long long), cudaMemcpyDeviceToHost);
            double sec = ms / 1000.0;
            std::cout << "TARGET[" << ti << "] '" << targets[ti] << "' ops=" << hops
                      << " time(s)=" << sec << " hashes/s=" << (sec > 0 ? (unsigned long long)(hops / sec) : 0ULL) << std::endl;
            if (h_found_idx != -1) {
                std::vector<char> h_found_pwd(MAX_PWD + 1);
                cudaMemcpy(h_found_pwd.data(), d_found_pwd, MAX_PWD + 1, cudaMemcpyDeviceToHost);
                int orig = -1;
                for (size_t k = 0; k < order.size(); ++k) {
                    int L = order[k];
                    if (h_found_idx >= attack_base[L] && h_found_idx < (long long)(attack_base[L] + bucket_count[L])) {
                        orig = h_origline[phys_base[L] + (int)(h_found_idx - attack_base[L])];
                        break;
                    }
                }
                if (orig < 0 && h_found_idx >= long_attack_base) orig = h_origline[long_base + (int)(h_found_idx - long_attack_base)];
                std::cout << "FOUND rank=" << h_found_idx << " fileline=" << orig << " pwd=" << std::string(h_found_pwd.data()) << std::endl;
            } else {
                std::cout << "NOT FOUND" << std::endl;
            }
        }
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cudaFree(d_target); cudaFree(d_found_idx); cudaFree(d_found_pwd); cudaFree(d_ops);
        cudaFree(d_flat); cudaFree(d_offsets); cudaFree(d_lens);
        cudaFreeHost(h_flat); cudaFreeHost(h_offsets); cudaFreeHost(h_lens);
        delete[] h_origline;
        return 0;
    }
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
    int L = 5;
    std::string password = "abcde";
    bool progress = false;
    std::vector<std::string> params;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; if (a == "--progress") progress = true; else params.push_back(a); }
    if (params.size() >= 1) L = std::stoi(params[0]);
    if (params.size() >= 2) alphabet = params[1];
    if (params.size() >= 3) password = params[2];
    if (alphabet == "abcdefghijklmnopqrstuvwxyz") {
        std::cout << "HINT: enumeration goes least-significant-digit first; pass a frequency-ordered alphabet for Markov-like speedup, e.g. \"etaoinshrdlucmfwypvbgkjqxz\"" << std::endl;
    }
    if (L < 1 || L > 32 || alphabet.size() > 64) { std::cerr << "invalid params" << std::endl; return 1; }
    uint32_t target[8];
    sha256_host(password.c_str(), (int)password.size(), target);
    uint64_t total = pow_uint64((uint64_t)alphabet.size(), L);
    char h_alphabet[64];
    for (size_t i = 0; i < alphabet.size(); ++i) h_alphabet[i] = alphabet[i];
    cudaMemcpyToSymbol(d_alphabet, h_alphabet, 64);
    cudaMemcpyToSymbol(d_K, K64, sizeof(K64));
    uint32_t* d_target;
    cudaMalloc(&d_target, 8 * sizeof(uint32_t));
    cudaMemcpy(d_target, target, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    long long* d_found_idx;
    cudaMalloc(&d_found_idx, sizeof(long long));
    long long h_found_idx = -1;
    cudaMemcpy(d_found_idx, &h_found_idx, sizeof(long long), cudaMemcpyHostToDevice);
    char* d_found_pwd;
    cudaMalloc(&d_found_pwd, (size_t)L + 1);
    int threads = 256;
    uint64_t block64 = (total + (uint64_t)threads - 1ULL) / (uint64_t)threads;
    int blocks = (block64 > 65535ULL) ? 65535 : (int)block64;
    if (blocks < 1) blocks = 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    unsigned long long* d_ops;
    cudaMalloc(&d_ops, sizeof(unsigned long long));
    unsigned long long zero = 0ULL;
    cudaMemcpy(d_ops, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (false) {}
    else if (!progress) {
        cudaEventRecord(start);
        dispatch_brute(L, (int)alphabet.size(), 0ULL, total, blocks, threads, d_target, d_found_idx, d_found_pwd, d_ops);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else {
        uint64_t chunk = total < 500000000ULL ? total : 500000000ULL;
        uint64_t processed = 0ULL;
        unsigned long long ops_total = 0ULL;
        float ms_total = 0.0f;
        while (processed < total) {
            uint64_t limit = (processed + chunk > total) ? (total - processed) : chunk;
            uint64_t block64c = (limit + (uint64_t)threads - 1ULL) / (uint64_t)threads;
            int blocksc = (block64c > 65535ULL) ? 65535 : (int)block64c;
            if (blocksc < 1) blocksc = 1;
            unsigned long long zero = 0ULL; cudaMemcpy(d_ops, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);
            cudaEventRecord(start);
            dispatch_brute(L, (int)alphabet.size(), processed, limit, blocksc, threads, d_target, d_found_idx, d_found_pwd, d_ops);
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float ms_chunk = 0.0f; cudaEventElapsedTime(&ms_chunk, start, stop);
            unsigned long long h_ops_chunk = 0ULL; cudaMemcpy(&h_ops_chunk, d_ops, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            ops_total += h_ops_chunk; ms_total += ms_chunk;
            cudaMemcpy(&h_found_idx, d_found_idx, sizeof(long long), cudaMemcpyDeviceToHost);
            if (h_found_idx != -1) break;
            processed += limit;
            std::cout << "PROGRESS ops=" << ops_total << " time(s)=" << (ms_total/1000.0f) << " hashes/s=" << (unsigned long long)(ops_total/(ms_total/1000.0f)) << std::endl;
        }
        double gpu_sec2 = ms_total / 1000.0;
        double gpu_hps2 = (gpu_sec2 > 0 ? (double)ops_total / gpu_sec2 : 0.0);
        std::vector<char> h_found_pwd2(L + 1);
        cudaMemcpy(h_found_pwd2.data(), d_found_pwd, (size_t)L + 1, cudaMemcpyDeviceToHost);
        std::cout << "GPU length=" << L << " alphabet=" << alphabet.size() << " total=" << total << std::endl;
        std::cout << "GPU ops=" << ops_total << " time(s)=" << gpu_sec2 << " hashes/s=" << (uint64_t)gpu_hps2 << std::endl;
        if (h_found_idx != -1) std::cout << "GPU found index=" << h_found_idx << " pwd=" << std::string(h_found_pwd2.data()) << std::endl; else std::cout << "GPU not found" << std::endl;
    }
    if (!progress) {
        float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop);
        cudaMemcpy(&h_found_idx, d_found_idx, sizeof(long long), cudaMemcpyDeviceToHost);
        unsigned long long h_ops = 0ULL; cudaMemcpy(&h_ops, d_ops, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        std::vector<char> h_found_pwd(L + 1);
        cudaMemcpy(h_found_pwd.data(), d_found_pwd, (size_t)L + 1, cudaMemcpyDeviceToHost);
        double gpu_sec = ms / 1000.0;
        double gpu_hps = (gpu_sec > 0 ? (double)h_ops / gpu_sec : 0.0);
        std::cout << "GPU length=" << L << " alphabet=" << alphabet.size() << " total=" << total << std::endl;
        std::cout << "GPU ops=" << h_ops << " time(s)=" << gpu_sec << " hashes/s=" << (uint64_t)gpu_hps << std::endl;
        if (h_found_idx != -1) std::cout << "GPU found index=" << h_found_idx << " pwd=" << std::string(h_found_pwd.data()) << std::endl; else std::cout << "GPU not found" << std::endl;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (total <= 200000000ULL) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        long long cpu_found = -1;
        std::string candidate;
        uint32_t hv[8];
        for (uint64_t idx = 0; idx < total; ++idx) {
            idx_to_string(idx, L, alphabet, candidate);
            sha256_host(candidate.c_str(), L, hv);
            bool eq = true;
            for (int i = 0; i < 8; ++i) { if (hv[i] != target[i]) { eq = false; break; } }
            if (eq) { cpu_found = (long long)idx; break; }
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_sec = std::chrono::duration<double>(cpu_end - cpu_start).count();
        double cpu_hps = (cpu_sec > 0 ? (double)(cpu_found != -1 ? (uint64_t)cpu_found + 1ULL : total) / cpu_sec : 0.0);
        std::cout << "CPU ops=" << (cpu_found != -1 ? (uint64_t)cpu_found + 1ULL : total) << " time(s)=" << cpu_sec << " hashes/s=" << (uint64_t)cpu_hps << std::endl;
        if (cpu_found != -1) std::cout << "CPU found index=" << cpu_found << std::endl; else std::cout << "CPU not found" << std::endl;
    } else {
        std::cout << "CPU baseline skipped (total>2e8)" << std::endl;
    }
    cudaFree(d_target);
    cudaFree(d_found_idx);
    cudaFree(d_found_pwd);
    cudaFree(d_ops);
    return 0;
}
