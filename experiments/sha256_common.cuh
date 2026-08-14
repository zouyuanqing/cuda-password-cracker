// Shared SHA-256 device core for optimization prototypes.
// Self-contained; do NOT modify brute_sha256.cu.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

__constant__ char d_alphabet[64];
__constant__ uint32_t d_K[64];

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

// ---- baseline compress (mirrors brute_sha256.cu) ----
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
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = hh + S1 + ch + d_K[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        hh = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

// ---- variant: fully unrolled with #pragma unroll ----
__device__ __forceinline__ void sha256_compress_u(uint32_t* h, const unsigned char* chunk) {
    uint32_t w[64];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        int j = i * 4;
        w[i] = ((uint32_t)chunk[j] << 24) | ((uint32_t)chunk[j + 1] << 16) | ((uint32_t)chunk[j + 2] << 8) | ((uint32_t)chunk[j + 3]);
    }
#pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
#pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = hh + S1 + ch + d_K[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        hh = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

// ---- fixed-length single-block path: L is a compile-time constant ----
// Same algorithm as sha256_device, but L known at compile time so the compiler
// can fully unroll padding / constant-fold w[2..15]. Assumes L <= 55.
template<int L>
__device__ __forceinline__ void sha256_fixed(const char* msg, uint32_t* out) {
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    uint32_t w[64];
    unsigned char chunk[64];
#pragma unroll
    for (int i = 0; i < L; ++i) chunk[i] = (unsigned char)msg[i];
    chunk[L] = 0x80;
#pragma unroll
    for (int i = L + 1; i < 56; ++i) chunk[i] = 0;
    const uint64_t bitlen = (uint64_t)L * 8ULL;
#pragma unroll
    for (int i = 0; i < 8; ++i) chunk[63 - i] = (unsigned char)((bitlen >> (8 * i)) & 0xFF);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        int j = i * 4;
        w[i] = ((uint32_t)chunk[j] << 24) | ((uint32_t)chunk[j + 1] << 16) | ((uint32_t)chunk[j + 2] << 8) | ((uint32_t)chunk[j + 3]);
    }
#pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
#pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = hh + S1 + ch + d_K[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        hh = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }
    out[0] = h[0] + a; out[1] = h[1] + b; out[2] = h[2] + c; out[3] = h[3] + d;
    out[4] = h[4] + e; out[5] = h[5] + f; out[6] = h[6] + g; out[7] = h[7] + hh;
}

// single-block fast path (len <= 55)
__device__ __forceinline__ void sha256_device(const char* msg, int len, uint32_t* out) {
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    unsigned char chunk[64];
    for (int i = 0; i < len; ++i) chunk[i] = (unsigned char)msg[i];
    chunk[len] = 0x80;
    for (int i = len + 1; i < 56; ++i) chunk[i] = 0;
    uint64_t bitlen = (uint64_t)len * 8ULL;
    for (int i = 0; i < 8; ++i) chunk[63 - i] = (unsigned char)((bitlen >> (8 * i)) & 0xFF);
    sha256_compress(h, chunk);
    out[0] = h[0]; out[1] = h[1]; out[2] = h[2]; out[3] = h[3];
    out[4] = h[4]; out[5] = h[5]; out[6] = h[6]; out[7] = h[7];
}

__device__ __forceinline__ void sha256_device_u(const char* msg, int len, uint32_t* out) {
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    unsigned char chunk[64];
    for (int i = 0; i < len; ++i) chunk[i] = (unsigned char)msg[i];
    chunk[len] = 0x80;
    for (int i = len + 1; i < 56; ++i) chunk[i] = 0;
    uint64_t bitlen = (uint64_t)len * 8ULL;
    for (int i = 0; i < 8; ++i) chunk[63 - i] = (unsigned char)((bitlen >> (8 * i)) & 0xFF);
    sha256_compress_u(h, chunk);
    out[0] = h[0]; out[1] = h[1]; out[2] = h[2]; out[3] = h[3];
    out[4] = h[4]; out[5] = h[5]; out[6] = h[6]; out[7] = h[7];
}
