// CUDA ZIP password cracker v2
//  - ZipCrypto: dict + multi-length block-ticket brute kernel
//  - WinZip AES (AE-2): PBKDF2-HMAC-SHA1(1000) device kernel + dict/brute + maker/verifier
//  - frequency-ordered charset presets
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include "crypto_host.h"

// ================= ZipCrypto (PKZIP) =================
__constant__ uint32_t d_crc[256];
__constant__ char d_alphabet[96];
__constant__ unsigned char d_ver[6][12];
__constant__ unsigned char d_check[6];
__constant__ int d_nzip = 6;

__device__ __forceinline__ uint32_t crc32_step(uint32_t c, unsigned char b) {
    return d_crc[(c ^ b) & 0xff] ^ (c >> 8);
}
__device__ __forceinline__ void update_keys(uint32_t& k0, uint32_t& k1, uint32_t& k2, unsigned char c) {
    k0 = crc32_step(k0, c);
    k1 = (k1 + (k0 & 0xff)) * 134775813u + 1u;
    k2 = crc32_step(k2, (unsigned char)(k1 >> 24));
}
__device__ __forceinline__ unsigned char ks_byte(uint32_t k2) {
    uint32_t t = k2 | 2;
    return (unsigned char)((t * (t ^ 1)) >> 8);
}
__device__ __forceinline__ bool zip_check_one(const char* pw, int len, int e) {
    uint32_t k0 = 0x12345678u, k1 = 0x23456789u, k2 = 0x34567890u;
    for (int i = 0; i < len; ++i) update_keys(k0, k1, k2, (unsigned char)pw[i]);
    for (int i = 0; i < 11; ++i) {
        unsigned char c = d_ver[e][i] ^ ks_byte(k2);
        update_keys(k0, k1, k2, c);
    }
    return (d_ver[e][11] ^ ks_byte(k2)) == d_check[e];
}
__device__ __forceinline__ bool zip_check_all(const char* pw, int len) {
    if (!zip_check_one(pw, len, 0)) return false;
    for (int e = 1; e < d_nzip; ++e)
        if (!zip_check_one(pw, len, e)) return false;
    return true;
}

__global__ void zip_dict_kernel(const char* flat, const int* offsets, const int* lens, int count,
                                long long* found_idx, char* found_pwd, unsigned long long* ops_counter) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    while (idx < (uint64_t)count) {
        if (*found_idx != -1) return;
        const char* msg = flat + offsets[(int)idx];
        int len = lens[(int)idx];
        atomicAdd(ops_counter, 1ULL);
        if (zip_check_all(msg, len)) {
            if (atomicCAS((unsigned long long*)found_idx, ~0ULL, (unsigned long long)idx) == ~0ULL) {
                for (int i = 0; i < len && i < 95; ++i) found_pwd[i] = msg[i];
                found_pwd[len < 95 ? len : 95] = '\0';
            }
            return;
        }
        idx += stride;
    }
}

// per-thread contiguous walk of a fixed-length candidate range (carry enumeration)
template<int L>
__device__ __forceinline__ bool zip_run_range(int N, uint64_t inner0, uint64_t count,
                                              long long* found_idx, int* found_len, char* found_pwd,
                                              unsigned long long* ops_counter) {
    unsigned int d[L];
    uint64_t t = inner0;
#pragma unroll
    for (int i = 0; i < L; ++i) { d[i] = (unsigned int)(t % (uint64_t)N); t /= (uint64_t)N; }
    char buf[L];
    for (uint64_t k = 0; k < count; ++k) {
        if (*found_idx != -1) return true;
#pragma unroll
        for (int i = 0; i < L; ++i) buf[i] = d_alphabet[d[i]];
        atomicAdd(ops_counter, 1ULL);
        if (zip_check_all(buf, L)) {
            if (atomicCAS((unsigned long long*)found_idx, ~0ULL, (unsigned long long)(inner0 + k)) == ~0ULL) {
#pragma unroll
                for (int i = 0; i < L; ++i) found_pwd[i] = buf[i];
                found_pwd[L] = '\0';
                *found_len = L;
            }
            return true;
        }
        unsigned int carry = 1u;
#pragma unroll
        for (int i = 0; i < L; ++i) {
            unsigned int s = d[i] + carry;
            if (s >= (unsigned int)N) { s -= (unsigned int)N; carry = 1u; } else { carry = 0u; }
            d[i] = s;
        }
    }
    return false;
}

// multi-length kernel: one launch covers minL..maxL; blocks pull contiguous work
// chunks via a global ticket (automatic load balance, zero gaps between lengths).
__global__ void zip_brute_multi_kernel(int N, int minL, int maxL, const uint64_t* seg, uint64_t total,
                                       uint64_t* ticket, long long* found_idx, int* found_len,
                                       char* found_pwd, unsigned long long* ops_counter) {
    __shared__ uint64_t s_start;
    while (true) {
        if (threadIdx.x == 0) s_start = atomicAdd(ticket, (uint64_t)blockDim.x * 4);
        __syncthreads();
        if (*found_idx != -1) return;
        uint64_t start = s_start;
        uint64_t end = start + (uint64_t)blockDim.x * 4;
        if (start >= total) return;
        if (end > total) end = total;
        int L = minL;
        while (L <= maxL && seg[L - minL + 1] <= start) ++L;
        if (L > maxL) continue;
        uint64_t seg_end = seg[L - minL + 1];
        uint64_t a_end = end < seg_end ? end : seg_end;
        uint64_t n_a = a_end - start;
        if (n_a > 0) {
            uint64_t per = (n_a + blockDim.x - 1) / blockDim.x;
            uint64_t my0 = start + (uint64_t)threadIdx.x * per;
            uint64_t my1 = my0 + per; if (my1 > a_end) my1 = a_end;
            if (my0 < my1) {
                switch (L) {
                    case 1:  if (zip_run_range<1>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 2:  if (zip_run_range<2>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 3:  if (zip_run_range<3>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 4:  if (zip_run_range<4>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 5:  if (zip_run_range<5>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 6:  if (zip_run_range<6>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 7:  if (zip_run_range<7>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 8:  if (zip_run_range<8>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 9:  if (zip_run_range<9>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 10: if (zip_run_range<10>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 11: if (zip_run_range<11>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 12: if (zip_run_range<12>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 13: if (zip_run_range<13>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 14: if (zip_run_range<14>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 15: if (zip_run_range<15>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 16: if (zip_run_range<16>(N, my0 - seg[0], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                }
            }
        }
        if (a_end < end) {
            int L2 = L + 1;
            if (L2 > maxL) continue;
            uint64_t n_b = end - a_end;
            uint64_t per2 = (n_b + blockDim.x - 1) / blockDim.x;
            uint64_t my0 = a_end + (uint64_t)threadIdx.x * per2;
            uint64_t my1 = my0 + per2; if (my1 > end) my1 = end;
            if (my0 < my1) {
                switch (L2) {
                    case 2:  if (zip_run_range<2>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 3:  if (zip_run_range<3>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 4:  if (zip_run_range<4>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 5:  if (zip_run_range<5>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 6:  if (zip_run_range<6>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 7:  if (zip_run_range<7>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 8:  if (zip_run_range<8>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 9:  if (zip_run_range<9>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 10: if (zip_run_range<10>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 11: if (zip_run_range<11>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 12: if (zip_run_range<12>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 13: if (zip_run_range<13>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 14: if (zip_run_range<14>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 15: if (zip_run_range<15>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                    case 16: if (zip_run_range<16>(N, my0 - seg[1], my1 - my0, found_idx, found_len, found_pwd, ops_counter)) return; break;
                }
            }
        }
    }
}

// ================= WinZip AES (AE-2) =================
__constant__ unsigned char d_salt[6][16];
__constant__ int d_saltlen[6];
__constant__ int d_dklen[6];
__constant__ unsigned char d_aesver[6][2];
__constant__ int d_naes = 1;

__device__ __forceinline__ void sha1_compress(uint32_t* h, const unsigned char* p) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i)
        w[i] = ((uint32_t)p[i*4] << 24) | ((uint32_t)p[i*4+1] << 16) | ((uint32_t)p[i*4+2] << 8) | p[i*4+3];
    for (int i = 16; i < 80; ++i) w[i] = ((w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16]) << 1) | ((w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16]) >> 31);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;
        if (i < 20) { f = (b & c) | ((~b) & d); k = 0x5a827999u; }
        else if (i < 40) { f = b ^ c ^ d; k = 0x6ed9eba1u; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdcu; }
        else { f = b ^ c ^ d; k = 0xca62c1d6u; }
        uint32_t tmp = ((a << 5) | (a >> 27)) + f + e + k + w[i];
        e = d; d = c; c = (b << 30) | (b >> 2); b = a; a = tmp;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
}
__device__ void sha1_full(const unsigned char* data, int len, unsigned char out[20]) {
    uint32_t h[5] = { 0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u, 0xc3d2e1f0u };
    int total = (len + 72) / 64;
    unsigned char chunk[64];
    for (int b = 0; b < total; ++b) {
        int base = b * 64;
        for (int i = 0; i < 64; ++i) {
            int pos = base + i;
            if (pos < len) chunk[i] = data[pos];
            else if (pos == len) chunk[i] = 0x80;
            else chunk[i] = 0;
        }
        if (b == total - 1) {
            uint64_t bits = (uint64_t)len * 8ULL;
            for (int i = 0; i < 8; ++i) chunk[63 - i] = (unsigned char)(bits >> (8 * i));
        }
        sha1_compress(h, chunk);
    }
    for (int i = 0; i < 5; ++i) {
        out[i*4] = (unsigned char)(h[i] >> 24); out[i*4+1] = (unsigned char)(h[i] >> 16);
        out[i*4+2] = (unsigned char)(h[i] >> 8); out[i*4+3] = (unsigned char)h[i];
    }
}
// data length <= 64 assumed (salt||idx or U); key arbitrary
__device__ void hmac_sha1_dev(const unsigned char* key, int keylen,
                              const unsigned char* data, int datalen, unsigned char out[20]) {
    unsigned char k[64];
    for (int i = 0; i < 64; ++i) k[i] = 0;
    if (keylen > 64) sha1_full(key, keylen, k);
    else for (int i = 0; i < keylen; ++i) k[i] = key[i];
    unsigned char ipad[64], opad[64], buf[128], tmp[20];
    for (int i = 0; i < 64; ++i) { ipad[i] = k[i] ^ 0x36; opad[i] = k[i] ^ 0x5c; }
    for (int i = 0; i < 64; ++i) buf[i] = ipad[i];
    for (int i = 0; i < datalen; ++i) buf[64 + i] = data[i];
    sha1_full(buf, 64 + datalen, tmp);
    for (int i = 0; i < 64; ++i) buf[i] = opad[i];
    for (int i = 0; i < 20; ++i) buf[64 + i] = tmp[i];
    sha1_full(buf, 84, out);
}
__device__ void pbkdf2_dev(const unsigned char* pw, int pwlen,
                           const unsigned char* salt, int saltlen, int iters,
                           unsigned char* dk, int dklen) {
    int blocks = (dklen + 19) / 20;
    for (int b = 1; b <= blocks; ++b) {
        unsigned char msg[24];
        for (int i = 0; i < saltlen; ++i) msg[i] = salt[i];
        msg[saltlen] = (unsigned char)(b >> 24); msg[saltlen+1] = (unsigned char)(b >> 16);
        msg[saltlen+2] = (unsigned char)(b >> 8); msg[saltlen+3] = (unsigned char)b;
        unsigned char u[20], t[20];
        hmac_sha1_dev(pw, pwlen, msg, saltlen + 4, u);
        for (int i = 0; i < 20; ++i) t[i] = u[i];
        for (int i = 1; i < iters; ++i) {
            hmac_sha1_dev(pw, pwlen, u, 20, u);
            for (int j = 0; j < 20; ++j) t[j] ^= u[j];
        }
        int n = dklen - (b - 1) * 20; if (n > 20) n = 20;
        for (int j = 0; j < n; ++j) dk[(b - 1) * 20 + j] = t[j];
    }
}
__device__ __forceinline__ bool aes_check(const char* pw, int len) {
    unsigned char dk[34];
    for (int e = 0; e < d_naes; ++e) {
        pbkdf2_dev((const unsigned char*)pw, len, d_salt[e], d_saltlen[e], 1000, dk, d_dklen[e]);
        if (dk[d_dklen[e] - 2] != d_aesver[e][0] || dk[d_dklen[e] - 1] != d_aesver[e][1]) return false;
    }
    return true;
}

// candidate-slot collection (AES verifier is only 16 bits per entry, so the
// kernel collects ALL passing candidates; the host does the final PBKDF2+HMAC check)
#define AES_CAND_CAP 4096

__global__ void aes_dict_kernel(const char* flat, const int* offsets, const int* lens, int count,
                                char* found_cands, int* cand_lens, int cand_cap, int* cand_count,
                                unsigned long long* ops_counter) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    while (idx < (uint64_t)count) {
        const char* msg = flat + offsets[(int)idx];
        int len = lens[(int)idx];
        atomicAdd(ops_counter, 1ULL);
        if (aes_check(msg, len)) {
            int slot = atomicAdd(cand_count, 1);
            if (slot < cand_cap) {
                int wl = len < 95 ? len : 95;
                for (int i = 0; i < wl; ++i) found_cands[slot * 96 + i] = msg[i];
                found_cands[slot * 96 + wl] = '\0';
                cand_lens[slot] = wl;
            }
        }
        idx += stride;
    }
}

template<int L>
__device__ __forceinline__ void aes_run_range(int N, uint64_t inner0, uint64_t count,
                                              char* found_cands, int* cand_lens, int cand_cap,
                                              int* cand_count, unsigned long long* ops_counter) {
    unsigned int d[L];
    uint64_t t = inner0;
#pragma unroll
    for (int i = 0; i < L; ++i) { d[i] = (unsigned int)(t % (uint64_t)N); t /= (uint64_t)N; }
    char buf[L];
    for (uint64_t k = 0; k < count; ++k) {
#pragma unroll
        for (int i = 0; i < L; ++i) buf[i] = d_alphabet[d[i]];
        atomicAdd(ops_counter, 1ULL);
        if (aes_check(buf, L)) {
            int slot = atomicAdd(cand_count, 1);
            if (slot < cand_cap) {
#pragma unroll
                for (int i = 0; i < L; ++i) found_cands[slot * 96 + i] = buf[i];
                found_cands[slot * 96 + L] = '\0';
                cand_lens[slot] = L;
            }
        }
        unsigned int carry = 1u;
#pragma unroll
        for (int i = 0; i < L; ++i) {
            unsigned int s = d[i] + carry;
            if (s >= (unsigned int)N) { s -= (unsigned int)N; carry = 1u; } else { carry = 0u; }
            d[i] = s;
        }
    }
}

__global__ void aes_brute_multi_kernel(int N, int minL, int maxL, const uint64_t* seg, uint64_t total,
                                       uint64_t* ticket, char* found_cands, int* cand_lens, int cand_cap,
                                       int* cand_count, unsigned long long* ops_counter) {
    __shared__ uint64_t s_start;
    while (true) {
        if (threadIdx.x == 0) s_start = atomicAdd(ticket, (uint64_t)blockDim.x * 2);
        __syncthreads();
        uint64_t start = s_start;
        uint64_t end = start + (uint64_t)blockDim.x * 2;
        if (start >= total) return;
        if (end > total) end = total;
        int L = minL;
        while (L <= maxL && seg[L - minL + 1] <= start) ++L;
        if (L > maxL) continue;
        uint64_t seg_end = seg[L - minL + 1];
        uint64_t a_end = end < seg_end ? end : seg_end;
        uint64_t n_a = a_end - start;
        if (n_a > 0) {
            uint64_t per = (n_a + blockDim.x - 1) / blockDim.x;
            uint64_t my0 = start + (uint64_t)threadIdx.x * per;
            uint64_t my1 = my0 + per; if (my1 > a_end) my1 = a_end;
            if (my0 < my1) {
                switch (L) {
                    case 1:  aes_run_range<1>(N, my0 - seg[0],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 2:  aes_run_range<2>(N, my0 - seg[0],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 3:  aes_run_range<3>(N, my0 - seg[0],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 4:  aes_run_range<4>(N, my0 - seg[0],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 5:  aes_run_range<5>(N, my0 - seg[0],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 6:  aes_run_range<6>(N, my0 - seg[0],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 7:  aes_run_range<7>(N, my0 - seg[0],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 8:  aes_run_range<8>(N, my0 - seg[0],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 9:  aes_run_range<9>(N, my0 - seg[0],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 10: aes_run_range<10>(N, my0 - seg[0], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 11: aes_run_range<11>(N, my0 - seg[0], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 12: aes_run_range<12>(N, my0 - seg[0], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 13: aes_run_range<13>(N, my0 - seg[0], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 14: aes_run_range<14>(N, my0 - seg[0], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 15: aes_run_range<15>(N, my0 - seg[0], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 16: aes_run_range<16>(N, my0 - seg[0], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                }
            }
        }
        if (a_end < end) {
            int L2 = L + 1;
            if (L2 > maxL) continue;
            uint64_t n_b = end - a_end;
            uint64_t per2 = (n_b + blockDim.x - 1) / blockDim.x;
            uint64_t my0 = a_end + (uint64_t)threadIdx.x * per2;
            uint64_t my1 = my0 + per2; if (my1 > end) my1 = end;
            if (my0 < my1) {
                switch (L2) {
                    case 2:  aes_run_range<2>(N, my0 - seg[1],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 3:  aes_run_range<3>(N, my0 - seg[1],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 4:  aes_run_range<4>(N, my0 - seg[1],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 5:  aes_run_range<5>(N, my0 - seg[1],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 6:  aes_run_range<6>(N, my0 - seg[1],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 7:  aes_run_range<7>(N, my0 - seg[1],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 8:  aes_run_range<8>(N, my0 - seg[1],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 9:  aes_run_range<9>(N, my0 - seg[1],  my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 10: aes_run_range<10>(N, my0 - seg[1], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 11: aes_run_range<11>(N, my0 - seg[1], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 12: aes_run_range<12>(N, my0 - seg[1], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 13: aes_run_range<13>(N, my0 - seg[1], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 14: aes_run_range<14>(N, my0 - seg[1], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 15: aes_run_range<15>(N, my0 - seg[1], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                    case 16: aes_run_range<16>(N, my0 - seg[1], my1 - my0, found_cands, cand_lens, cand_cap, cand_count, ops_counter); break;
                }
            }
        }
    }
}

// ================= host helpers =================
static uint64_t pow_u64(uint64_t b, int e) { uint64_t r = 1; for (int i = 0; i < e; ++i) r *= b; return r; }

static void put16(std::vector<unsigned char>& v, uint16_t x) { v.push_back(x & 0xff); v.push_back(x >> 8); }
static void put32(std::vector<unsigned char>& v, uint32_t x) {
    v.push_back(x & 0xff); v.push_back((x >> 8) & 0xff); v.push_back((x >> 16) & 0xff); v.push_back((x >> 24) & 0xff);
}

struct AesEntry {
    std::string name;
    int strength;            // 1=128, 2=192, 3=256
    int dklen;               // keybytes + 2
    unsigned char salt[16];
    int saltlen;
    unsigned char verifier[2];
    uint64_t data_off;       // offset of ciphertext
    uint64_t data_len;       // ciphertext length
    uint64_t hmac_off;       // AE-2: 10-byte HMAC at end
    int actual_method;
    uint16_t flags;
};

static bool parse_aes_zip(const std::vector<unsigned char>& d, std::vector<AesEntry>& out) {
    out.clear();
    size_t p = 0;
    while (p + 30 <= d.size()) {
        if (d[p] == 0x50 && d[p+1] == 0x4b && d[p+2] == 0x03 && d[p+3] == 0x04) {
            uint16_t flags = d[p+6] | (d[p+7] << 8);
            uint16_t method = d[p+8] | (d[p+9] << 8);
            uint16_t namelen = d[p+26] | (d[p+27] << 8);
            uint16_t extralen = d[p+28] | (d[p+29] << 8);
            uint32_t csize = d[p+18] | (d[p+19]<<8) | (d[p+20]<<16) | ((uint32_t)d[p+21]<<24);
            if (method == 99 && (flags & 1)) {
                AesEntry e;
                e.name = std::string((const char*)&d[p+30], namelen);
                size_t x = p + 30 + namelen;
                // walk extra fields for 0x9901
                size_t xe = x + extralen;
                bool ok = false;
                while (x + 4 <= xe) {
                    uint16_t id = d[x] | (d[x+1] << 8);
                    uint16_t sz = d[x+2] | (d[x+3] << 8);
                    if (id == 0x9901 && sz >= 7) {
                        uint16_t ver = d[x+4] | (d[x+5] << 8);
                        e.strength = d[x+8];
                        e.actual_method = d[x+9] | (d[x+10] << 8);
                        if (ver != 2) { std::cerr << "only AE-2 supported (version " << ver << ")" << std::endl; return false; }
                        ok = true;
                    }
                    x += 4 + sz;
                }
                if (!ok) { std::cerr << "AES extra field not found" << std::endl; return false; }
                e.saltlen = (e.strength == 3) ? 16 : 8;
                e.dklen = (e.strength == 1 ? 16 : e.strength == 2 ? 24 : 32) + 2;
                x = p + 30 + namelen + extralen;
                memcpy(e.salt, &d[x], e.saltlen);
                memcpy(e.verifier, &d[x + e.saltlen], 2);
                e.data_off = x + e.saltlen + 2;
                e.data_len = csize;
                e.hmac_off = e.data_off + csize;
                e.flags = flags;
                out.push_back(e);
            }
            p = p + 30 + namelen + extralen + csize;
            continue;
        }
        if (d[p] == 0x50 && d[p+1] == 0x4b && (d[p+2] == 0x01 || d[p+2] == 0x05 || d[p+2] == 0x06 || d[p+2] == 0x07)) break;
        ++p;
    }
    return !out.empty();
}

static void load_charset(const std::string& name, std::string& charset) {
    if (name == "lower") charset = "abcdefghijklmnopqrstuvwxyz";
    else if (name == "digits") charset = "0123456789";
    else if (name == "lowerdigits") charset = "abcdefghijklmnopqrstuvwxyz0123456789";
    else if (name == "freq") charset = "1234567890etaoinshrdlucmfwypvbgkjqxz";
    else if (name == "mixed") charset = "1234567890etaoinshrdlucmfwypvbgkjqxzEATOINSHRDLUCMFWYPVBGKJQXZ";
    else charset = name;
}

struct ZipEntry { unsigned char ver[12]; unsigned char check; };

static bool parse_zipcrypto_zip(const std::vector<unsigned char>& d, std::vector<ZipEntry>& out) {
    out.clear();
    size_t p = 0;
    while (p + 30 <= d.size()) {
        if (d[p] == 0x50 && d[p+1] == 0x4b && d[p+2] == 0x03 && d[p+3] == 0x04) {
            uint16_t flags = d[p+6] | (d[p+7] << 8);
            uint16_t method = d[p+8] | (d[p+9] << 8);
            uint16_t namelen = d[p+26] | (d[p+27] << 8);
            uint16_t extralen = d[p+28] | (d[p+29] << 8);
            uint32_t csize = d[p+18] | (d[p+19]<<8) | (d[p+20]<<16) | ((uint32_t)d[p+21]<<24);
            if ((flags & 1) && method != 99) {
                ZipEntry e;
                size_t v = p + 30 + namelen + extralen;
                if (v + 12 <= d.size()) {
                    memcpy(e.ver, &d[v], 12);
                    if (flags & 0x08) {   // data descriptor present -> check byte = high byte of mod time
                        e.check = d[p+11];
                    } else {              // check byte = high byte of CRC32
                        e.check = d[p+17];
                    }
                    out.push_back(e);
                    if (out.size() >= 6) return true;
                }
            }
            p = p + 30 + namelen + extralen + csize;
            continue;
        }
        if (d[p] == 0x50 && d[p+1] == 0x4b && (d[p+2] == 0x01 || d[p+2] == 0x05 || d[p+2] == 0x06 || d[p+2] == 0x07)) break;
        ++p;
    }
    return !out.empty();
}

static void init_zipcrypto(const std::string& charset, const std::vector<ZipEntry>& entries) {
    uint32_t crc[256];
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
        crc[i] = c;
    }
    cudaMemcpyToSymbol(d_crc, crc, sizeof(crc));
    char alpha[96] = {0};
    for (size_t i = 0; i < charset.size() && i < 95; ++i) alpha[i] = charset[i];
    cudaMemcpyToSymbol(d_alphabet, alpha, sizeof(alpha));
    unsigned char ver[6][12] = {{0}};
    unsigned char chk[6] = {0};
    int n = (int)entries.size();
    if (n > 6) n = 6;
    for (int i = 0; i < n; ++i) {
        memcpy(ver[i], entries[i].ver, 12);
        chk[i] = entries[i].check;
    }
    cudaMemcpyToSymbol(d_ver, ver, sizeof(ver));
    cudaMemcpyToSymbol(d_check, chk, sizeof(chk));
    cudaMemcpyToSymbol(d_nzip, &n, sizeof(int));
}

static void init_aes(const std::vector<AesEntry>& entries) {
    unsigned char salt[6][16] = {{0}};
    int saltlen[6] = {0}, dklen[6] = {0};
    unsigned char ver2[6][2] = {{0}};
    int n = (int)entries.size();
    if (n > 6) n = 6;
    for (int i = 0; i < n; ++i) {
        memcpy(salt[i], entries[i].salt, entries[i].saltlen);
        saltlen[i] = entries[i].saltlen;
        dklen[i] = entries[i].dklen;
        memcpy(ver2[i], entries[i].verifier, 2);
    }
    int naes = n < 2 ? n : 2;   // verify 2 entries -> 32-bit filter
    if (naes < 1) naes = 1;
    cudaMemcpyToSymbol(d_salt, salt, sizeof(salt));
    cudaMemcpyToSymbol(d_saltlen, saltlen, sizeof(saltlen));
    cudaMemcpyToSymbol(d_dklen, dklen, sizeof(dklen));
    cudaMemcpyToSymbol(d_aesver, ver2, sizeof(ver2));
    cudaMemcpyToSymbol(d_naes, &naes, sizeof(int));
}

// ---- selftest kernel: PBKDF2 RFC 6070 vectors on device ----
__global__ void selftest_kernel(unsigned char* out) {
    const unsigned char pw[8] = { 'p','a','s','s','w','o','r','d' };
    const unsigned char salt[4] = { 's','a','l','t' };
    unsigned char dk[20];
    pbkdf2_dev(pw, 8, salt, 4, 1000, dk, 20);
    for (int i = 0; i < 20; ++i) out[i] = dk[i];
    pbkdf2_dev(pw, 8, salt, 4, 4096, dk, 20);
    for (int i = 0; i < 20; ++i) out[20 + i] = dk[i];
}

static bool hexcmp(const unsigned char* a, const char* hex, int n) {
    for (int i = 0; i < n; ++i) {
        unsigned v;
        char c = hex[i*2];
        v = (c <= '9') ? (c - '0') : (c - 'a' + 10);
        c = hex[i*2+1];
        v = v * 16 + ((c <= '9') ? (c - '0') : (c - 'a' + 10));
        if (a[i] != v) return false;
    }
    return true;
}

static int do_selftest() {
    int fails = 0;
    const unsigned char pw[8] = { 'p','a','s','s','w','o','r','d' };
    const unsigned char salt[4] = { 's','a','l','t' };
    unsigned char dk[20];
    pbkdf2_hmac_sha1(pw, 8, salt, 4, 1, dk, 20);
    bool ok1 = hexcmp(dk, "0c60c80f961f0e71f3a9b524af6012062fe037a6", 20);
    pbkdf2_hmac_sha1(pw, 8, salt, 4, 1000, dk, 20);
    bool ok1000 = hexcmp(dk, "6e88be8bad7eae9d9e10aa061224034fed48d03f", 20);
    pbkdf2_hmac_sha1(pw, 8, salt, 4, 4096, dk, 20);
    bool ok4096 = hexcmp(dk, "4b007901b765489abead49d926f721d065a429c1", 20);
    // HMAC-SHA1 vector
    unsigned char hm[20];
    const unsigned char hkey[3] = { 'k','e','y' };
    const unsigned char hdata[] = "The quick brown fox jumps over the lazy dog";
    hmac_sha1(hkey, 3, hdata, 43, hm);
    bool okhmac = hexcmp(hm, "de7c9b85b8b78aa6bc8a7a36f70a90701c9db4d9", 20);
    // AES-128 FIPS-197 vector
    AesKey ak;
    unsigned char akey[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    unsigned char apt[16] = {0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff};
    unsigned char act[16];
    aes_key_expand(ak, akey, 128);
    aes_encrypt_block(ak, apt, act);
    bool okaes = hexcmp(act, "69c4e0d86a7b0430d8cdb78070b4c55a", 16);
    std::cout << "HOST  PBKDF2 iter=1    " << (ok1 ? "PASS" : "FAIL") << std::endl;
    std::cout << "HOST  PBKDF2 iter=1000 " << (ok1000 ? "PASS" : "FAIL") << std::endl;
    std::cout << "HOST  PBKDF2 iter=4096 " << (ok4096 ? "PASS" : "FAIL") << std::endl;
    std::cout << "HOST  HMAC-SHA1        " << (okhmac ? "PASS" : "FAIL") << std::endl;
    std::cout << "HOST  AES-128 (FIPS)   " << (okaes ? "PASS" : "FAIL") << std::endl;
    if (!ok1 || !ok1000 || !ok4096 || !okhmac || !okaes) fails++;
    // device
    unsigned char* d_out; cudaMalloc(&d_out, 40);
    unsigned char h_out[40] = {0};
    cudaMemcpy(d_out, h_out, 40, cudaMemcpyHostToDevice);
    selftest_kernel<<<1, 1>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 40, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    bool dok1 = hexcmp(h_out, "6e88be8bad7eae9d9e10aa061224034fed48d03f", 20);
    bool dok1000 = hexcmp(h_out + 20, "4b007901b765489abead49d926f721d065a429c1", 20);
    std::cout << "DEVICE PBKDF2 iter=1000 " << (dok1 ? "PASS" : "FAIL") << std::endl;
    std::cout << "DEVICE PBKDF2 iter=4096 " << (dok1000 ? "PASS" : "FAIL") << std::endl;
    if (!dok1 || !dok1000) fails++;
    return fails;
}

// ---- run one multi-length brute scan (ZipCrypto) ----
static bool run_multi_brute(const std::string& charset, int minL, int maxL,
                            std::string& out_pwd, int& out_L, unsigned long long& ops, double& sec) {
    int N = (int)charset.size();
    std::vector<uint64_t> seg(maxL - minL + 2);
    seg[0] = 0;
    for (int L = minL; L <= maxL; ++L) seg[L - minL + 1] = seg[L - minL] + pow_u64(N, L);
    uint64_t total = seg[maxL - minL + 1];
    uint64_t* d_seg; cudaMalloc(&d_seg, seg.size() * sizeof(uint64_t));
    cudaMemcpy(d_seg, seg.data(), seg.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    uint64_t* d_ticket; cudaMalloc(&d_ticket, sizeof(uint64_t));
    uint64_t zero = 0;
    cudaMemcpy(d_ticket, &zero, sizeof(uint64_t), cudaMemcpyHostToDevice);
    long long* d_found_idx; cudaMalloc(&d_found_idx, sizeof(long long));
    long long nf = -1;
    cudaMemcpy(d_found_idx, &nf, sizeof(long long), cudaMemcpyHostToDevice);
    int* d_found_len; cudaMalloc(&d_found_len, sizeof(int));
    char* d_found_pwd; cudaMalloc(&d_found_pwd, 96);
    unsigned long long* d_ops; cudaMalloc(&d_ops, sizeof(unsigned long long));
    cudaMemcpy(d_ops, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = 65535;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    zip_brute_multi_kernel<<<blocks, threads>>>(N, minL, maxL, d_seg, total, d_ticket, d_found_idx, d_found_len, d_found_pwd, d_ops);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    sec = ms / 1000.0;
    cudaMemcpy(&ops, d_ops, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    long long h_found;
    cudaMemcpy(&h_found, d_found_idx, sizeof(long long), cudaMemcpyDeviceToHost);
    int h_len = 0;
    cudaMemcpy(&h_len, d_found_len, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_found != -1) {
        std::vector<char> pwd(96);
        cudaMemcpy(pwd.data(), d_found_pwd, 96, cudaMemcpyDeviceToHost);
        out_pwd = std::string(pwd.data());
        out_L = h_len;
    }
    cudaFree(d_seg); cudaFree(d_ticket); cudaFree(d_found_idx); cudaFree(d_found_len); cudaFree(d_found_pwd); cudaFree(d_ops);
    return h_found != -1;
}

// ---- AES zip maker (AE-2, stored) ----
static int do_aes_make(const std::string& path, const std::string& pwd, const std::string& content) {
    int strength = 3;   // 256-bit
    int keybytes = 32;
    unsigned char salt[16];
    {   // deterministic pseudo-random salt from clock
        auto ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        for (int i = 0; i < 16; ++i) salt[i] = (unsigned char)((ns >> (i * 5)) ^ (i * 131));
    }
    unsigned char dk[34];
    pbkdf2_hmac_sha1((const unsigned char*)pwd.data(), pwd.size(), salt, 16, 1000, dk, keybytes + 2);
    unsigned char verifier[2] = { dk[keybytes], dk[keybytes+1] };
    AesKey ak;
    aes_key_expand(ak, dk, keybytes * 8);
    // ciphertext = CTR(counter=1) over content; HMAC over verifier||ciphertext
    std::vector<unsigned char> ct(content.begin(), content.end());
    unsigned char counter[16] = {0};
    counter[15] = 1;
    aes_ctr_crypt(ak, ct.data(), ct.size(), counter);
    std::vector<unsigned char> hmac_in;
    hmac_in.insert(hmac_in.end(), verifier, verifier + 2);
    hmac_in.insert(hmac_in.end(), ct.begin(), ct.end());
    unsigned char hm[20];
    hmac_sha1(dk, keybytes, hmac_in.data(), hmac_in.size(), hm);
    // build zip
    std::vector<unsigned char> z;
    std::string name = "secret.txt";
    uint16_t flags = 0x0001;
    uint16_t method = 99;
    uint32_t crc = 0;               // AE-2: zero
    uint32_t csize = (uint32_t)ct.size();
    uint32_t usize = (uint32_t)content.size();
    // extra 0x9901: version 2, vendor "AE", strength, actual method (stored=0)
    std::vector<unsigned char> extra;
    put16(extra, 0x9901); put16(extra, 7);
    extra.push_back(2); extra.push_back(0);   // version 2
    extra.push_back('A'); extra.push_back('E');
    extra.push_back((unsigned char)strength);
    put16(extra, 0);                          // actual method: stored
    // local header
    put32(z, 0x04034b50);
    put16(z, 51); put16(z, flags);
    put16(z, method);
    put16(z, 0); put16(z, 0x21);              // time/date
    put32(z, crc);
    put32(z, csize); put32(z, usize);
    put16(z, (uint16_t)name.size()); put16(z, (uint16_t)extra.size());
    z.insert(z.end(), name.begin(), name.end());
    z.insert(z.end(), extra.begin(), extra.end());
    z.insert(z.end(), salt, salt + 16);
    z.push_back(verifier[0]); z.push_back(verifier[1]);
    z.insert(z.end(), ct.begin(), ct.end());
    z.insert(z.end(), hm, hm + 10);
    uint32_t local_off = 0;
    // central directory
    uint32_t cd_off = (uint32_t)z.size();
    put32(z, 0x02014b50);
    put16(z, 51); put16(z, 51);
    put16(z, flags); put16(z, method);
    put16(z, 0); put16(z, 0x21);
    put32(z, crc);
    put32(z, csize); put32(z, usize);
    put16(z, (uint16_t)name.size()); put16(z, (uint16_t)extra.size());
    put16(z, 0);                    // comment
    put16(z, 0); put16(z, 0);       // disk, internal attrs
    put32(z, 0);                    // external attrs
    put32(z, local_off);
    z.insert(z.end(), name.begin(), name.end());
    z.insert(z.end(), extra.begin(), extra.end());
    uint32_t cd_size = (uint32_t)(z.size() - cd_off);
    // EOCD
    put32(z, 0x06054b50);
    put16(z, 0); put16(z, 0);
    put16(z, 1); put16(z, 1);
    put32(z, cd_size);
    put32(z, cd_off);
    put16(z, 0);
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)z.data(), z.size());
    f.close();
    std::cout << "wrote " << path << " (" << z.size() << " bytes, AES-" << keybytes*8
              << ", content=" << content.size() << " bytes)" << std::endl;
    return 0;
}

// ---- AES verify: PBKDF2 + HMAC check + decrypt to raw ----
static int do_aes_verify(const std::string& path, const std::string& pwd, const std::string& outdir) {
    std::ifstream f(path, std::ios::binary);
    std::vector<unsigned char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<AesEntry> entries;
    if (!parse_aes_zip(d, entries)) { std::cerr << "no AES entries" << std::endl; return 1; }
    int ok = 0;
    for (auto& e : entries) {
        int keybytes = e.dklen - 2;
        unsigned char dk[34];
        pbkdf2_hmac_sha1((const unsigned char*)pwd.data(), pwd.size(), e.salt, e.saltlen, 1000, dk, e.dklen);
        if (dk[keybytes] != e.verifier[0] || dk[keybytes+1] != e.verifier[1]) {
            std::cout << "FAIL " << e.name << " (verifier mismatch)" << std::endl;
            continue;
        }
        // HMAC over verifier || ciphertext
        std::vector<unsigned char> hmac_in;
        hmac_in.insert(hmac_in.end(), e.verifier, e.verifier + 2);
        hmac_in.insert(hmac_in.end(), d.begin() + e.data_off, d.begin() + e.data_off + e.data_len);
        unsigned char hm[20];
        hmac_sha1(dk, keybytes, hmac_in.data(), hmac_in.size(), hm);
        bool hmac_ok = (memcmp(hm, &d[e.hmac_off], 10) == 0);
        std::cout << (hmac_ok ? "OK  " : "FAIL") << " " << e.name << " verifier+hmac=" << (hmac_ok ? "match" : "MISMATCH") << std::endl;
        if (hmac_ok) {
            AesKey ak;
            aes_key_expand(ak, dk, keybytes * 8);
            std::vector<unsigned char> ct(d.begin() + e.data_off, d.begin() + e.data_off + e.data_len);
            unsigned char counter[16] = {0};
            counter[15] = 1;
            aes_ctr_crypt(ak, ct.data(), ct.size(), counter);
            std::string outname = outdir + "\\" + e.name.substr(e.name.find_last_of("/\\") + 1);
            std::ofstream fo(outname, std::ios::binary);
            fo.write((const char*)ct.data(), ct.size());
            fo.close();
            std::cout << "     decrypted -> " << outname << " (" << ct.size() << " bytes)" << std::endl;
            ok++;
        }
    }
    std::cout << "verified " << ok << "/" << entries.size() << " entries" << std::endl;
    return ok == (int)entries.size() ? 0 : 1;
}

int main(int argc, char** argv) {
    std::string mode = argc >= 2 ? argv[1] : "";
    if (mode == "--selftest") { return do_selftest(); }

    if (mode == "dict" && argc >= 4) {
        std::string zippath = argv[2];
        std::string path = argv[3];
        std::ifstream zf(zippath, std::ios::binary);
        if (!zf) { std::cerr << "open zip failed" << std::endl; return 1; }
        std::vector<unsigned char> zd((std::istreambuf_iterator<char>(zf)), std::istreambuf_iterator<char>());
        std::vector<ZipEntry> zents;
        if (!parse_zipcrypto_zip(zd, zents)) { std::cerr << "no ZipCrypto entries found" << std::endl; return 1; }
        std::cout << "zip entries=" << zents.size() << std::endl;
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) { std::cerr << "open dict failed" << std::endl; return 1; }
        std::streamoff fsize = in.tellg();
        in.seekg(0, std::ios::beg);
        init_zipcrypto("", zents);
        std::vector<char> flat((size_t)fsize);
        std::vector<int> offsets, lens;
        in.read(flat.data(), fsize);
        in.close();
        {
            long long i = 0;
            while (i < fsize) {
                long long s = i;
                while (i < fsize && flat[i] != '\n') ++i;
                long long e = i;
                if (e > s && flat[e - 1] == '\r') e -= 1;
                int len = (int)(e - s);
                if (len >= 1 && len <= 95) { offsets.push_back((int)s); lens.push_back(len); }
                if (i < fsize) ++i;
            }
        }
        int count = (int)offsets.size();
        std::cout << "dict lines=" << count << std::endl;
        char* d_flat; cudaMalloc(&d_flat, (size_t)fsize); cudaMemcpy(d_flat, flat.data(), (size_t)fsize, cudaMemcpyHostToDevice);
        int* d_offsets; cudaMalloc(&d_offsets, count * sizeof(int)); cudaMemcpy(d_offsets, offsets.data(), count * sizeof(int), cudaMemcpyHostToDevice);
        int* d_lens; cudaMalloc(&d_lens, count * sizeof(int)); cudaMemcpy(d_lens, lens.data(), count * sizeof(int), cudaMemcpyHostToDevice);
        long long* d_found_idx; cudaMalloc(&d_found_idx, sizeof(long long));
        long long nf = -1; cudaMemcpy(d_found_idx, &nf, sizeof(long long), cudaMemcpyHostToDevice);
        char* d_found_pwd; cudaMalloc(&d_found_pwd, 96);
        unsigned long long* d_ops; cudaMalloc(&d_ops, sizeof(unsigned long long));
        unsigned long long zero = 0; cudaMemcpy(d_ops, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (count + threads - 1) / threads; if (blocks > 65535) blocks = 65535; if (blocks < 1) blocks = 1;
        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        zip_dict_kernel<<<blocks, threads>>>(d_flat, d_offsets, d_lens, count, d_found_idx, d_found_pwd, d_ops);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        unsigned long long ops = 0; cudaMemcpy(&ops, d_ops, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        long long h_found; cudaMemcpy(&h_found, d_found_idx, sizeof(long long), cudaMemcpyDeviceToHost);
        double sec = ms / 1000.0;
        std::cout << "ops=" << ops << " time(s)=" << sec << " cands/s=" << (sec > 0 ? (unsigned long long)(ops / sec) : 0ULL) << std::endl;
        if (h_found != -1) {
            std::vector<char> pwd(96);
            cudaMemcpy(pwd.data(), d_found_pwd, 96, cudaMemcpyDeviceToHost);
            std::cout << "FOUND pwd='" << std::string(pwd.data()) << "' idx=" << h_found << std::endl;
        } else std::cout << "NOT FOUND" << std::endl;
        return 0;
    }
    if (mode == "brute" && argc >= 6) {
        std::string zippath = argv[2];
        std::ifstream zf(zippath, std::ios::binary);
        if (!zf) { std::cerr << "open zip failed" << std::endl; return 1; }
        std::vector<unsigned char> zd((std::istreambuf_iterator<char>(zf)), std::istreambuf_iterator<char>());
        std::vector<ZipEntry> zents;
        if (!parse_zipcrypto_zip(zd, zents)) { std::cerr << "no ZipCrypto entries found" << std::endl; return 1; }
        std::cout << "zip entries=" << zents.size() << std::endl;
        std::string cs;
        load_charset(argv[3], cs);
        int minL = std::stoi(argv[4]);
        int maxL = std::stoi(argv[5]);
        init_zipcrypto(cs, zents);
        std::cout << "charset=" << cs << " (" << cs.size() << " chars) L=" << minL << ".." << maxL << std::endl;
        std::string pwd; int L = 0; unsigned long long ops = 0; double sec = 0;
        bool found = run_multi_brute(cs, minL, maxL, pwd, L, ops, sec);
        std::cout << "ops=" << ops << " time(s)=" << sec << " cands/s=" << (sec > 0 ? (unsigned long long)(ops / sec) : 0ULL) << std::endl;
        if (found) std::cout << "FOUND pwd='" << pwd << "' L=" << L << std::endl;
        else std::cout << "NOT FOUND" << std::endl;
        return 0;
    }
    if (mode == "aes-make" && argc >= 5) {
        return do_aes_make(argv[2], argv[3], argv[4]);
    }
    if (mode == "aes-verify" && argc >= 4) {
        std::string outdir = argc >= 5 ? argv[4] : ".";
        return do_aes_verify(argv[2], argv[3], outdir);
    }
    if ((mode == "aes-dict" || mode == "aes-brute") && argc >= 4) {
        // parse zip -> AES entries -> init device
        std::ifstream f(argv[2], std::ios::binary);
        if (!f) { std::cerr << "open zip failed" << std::endl; return 1; }
        std::vector<unsigned char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::vector<AesEntry> entries;
        if (!parse_aes_zip(d, entries)) { std::cerr << "no AE-2 entries found" << std::endl; return 1; }
        init_aes(entries);
        std::cout << "AES entries=" << entries.size() << " strength=" << entries[0].strength << std::endl;
        // host-side final check: PBKDF2 verifier + AE-2 HMAC over the file's ciphertext
        auto verify_pwd = [&](const std::string& pwd) -> bool {
            for (auto& e : entries) {
                int keybytes = e.dklen - 2;
                unsigned char dk[34];
                pbkdf2_hmac_sha1((const unsigned char*)pwd.data(), pwd.size(), e.salt, e.saltlen, 1000, dk, e.dklen);
                if (dk[keybytes] != e.verifier[0] || dk[keybytes+1] != e.verifier[1]) return false;
                std::vector<unsigned char> hi;
                hi.insert(hi.end(), e.verifier, e.verifier + 2);
                hi.insert(hi.end(), d.begin() + e.data_off, d.begin() + e.data_off + e.data_len);
                unsigned char hm[20];
                hmac_sha1(dk, keybytes, hi.data(), hi.size(), hm);
                if (memcmp(hm, &d[e.hmac_off], 10) != 0) return false;
            }
            return true;
        };
        char* d_cands; cudaMalloc(&d_cands, (size_t)AES_CAND_CAP * 96);
        int* d_cand_lens; cudaMalloc(&d_cand_lens, AES_CAND_CAP * sizeof(int));
        int* d_cand_count; cudaMalloc(&d_cand_count, sizeof(int));
        unsigned long long* d_ops; cudaMalloc(&d_ops, sizeof(unsigned long long));
        unsigned long long ops = 0;
        int cand_cnt = 0;
        double sec = 0.0;
        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        if (mode == "aes-dict") {
            std::ifstream in(argv[3], std::ios::binary | std::ios::ate);
            if (!in) { std::cerr << "open dict failed" << std::endl; return 1; }
            std::streamoff fsize = in.tellg();
            in.seekg(0, std::ios::beg);
            std::vector<char> flat((size_t)fsize);
            std::vector<int> offsets, lens;
            in.read(flat.data(), fsize);
            in.close();
            long long i = 0;
            while (i < fsize) {
                long long s = i;
                while (i < fsize && flat[i] != '\n') ++i;
                long long e = i;
                if (e > s && flat[e - 1] == '\r') e -= 1;
                int len = (int)(e - s);
                if (len >= 1 && len <= 95) { offsets.push_back((int)s); lens.push_back(len); }
                if (i < fsize) ++i;
            }
            int count = (int)offsets.size();
            std::cout << "dict lines=" << count << std::endl;
            char* d_flat; cudaMalloc(&d_flat, (size_t)fsize); cudaMemcpy(d_flat, flat.data(), (size_t)fsize, cudaMemcpyHostToDevice);
            int* d_offsets; cudaMalloc(&d_offsets, count * sizeof(int)); cudaMemcpy(d_offsets, offsets.data(), count * sizeof(int), cudaMemcpyHostToDevice);
            int* d_lens; cudaMalloc(&d_lens, count * sizeof(int)); cudaMemcpy(d_lens, lens.data(), count * sizeof(int), cudaMemcpyHostToDevice);
            int zero = 0;
            cudaMemcpy(d_cand_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
            unsigned long long z2 = 0;
            cudaMemcpy(d_ops, &z2, sizeof(unsigned long long), cudaMemcpyHostToDevice);
            int threads = 256;
            int blocks = (count + threads - 1) / threads; if (blocks > 65535) blocks = 65535; if (blocks < 1) blocks = 1;
            cudaEventRecord(start);
            aes_dict_kernel<<<blocks, threads>>>(d_flat, d_offsets, d_lens, count, d_cands, d_cand_lens, AES_CAND_CAP, d_cand_count, d_ops);
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop);
            sec = ms / 1000.0;
            cudaMemcpy(&ops, d_ops, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            cudaMemcpy(&cand_cnt, d_cand_count, sizeof(int), cudaMemcpyDeviceToHost);
        } else {
            std::string cs;
            load_charset(argv[3], cs);
            int minL = std::stoi(argv[4]);
            int maxL = std::stoi(argv[5]);
            std::vector<ZipEntry> noz;
            init_zipcrypto(cs, noz);   // sets d_alphabet (needed by the brute kernels)
            std::cout << "charset=" << cs << " (" << cs.size() << " chars) L=" << minL << ".." << maxL << std::endl;
            int N = (int)cs.size();
            std::vector<uint64_t> seg(maxL - minL + 2);
            seg[0] = 0;
            for (int L = minL; L <= maxL; ++L) seg[L - minL + 1] = seg[L - minL] + pow_u64(N, L);
            uint64_t total = seg[maxL - minL + 1];
            uint64_t* d_seg; cudaMalloc(&d_seg, seg.size() * sizeof(uint64_t));
            cudaMemcpy(d_seg, seg.data(), seg.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
            uint64_t* d_ticket; cudaMalloc(&d_ticket, sizeof(uint64_t));
            uint64_t tz = 0;
            cudaMemcpy(d_ticket, &tz, sizeof(uint64_t), cudaMemcpyHostToDevice);
            int zero = 0;
            cudaMemcpy(d_cand_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
            unsigned long long z2 = 0;
            cudaMemcpy(d_ops, &z2, sizeof(unsigned long long), cudaMemcpyHostToDevice);
            cudaEventRecord(start);
            aes_brute_multi_kernel<<<65535, 256>>>(N, minL, maxL, d_seg, total, d_ticket, d_cands, d_cand_lens, AES_CAND_CAP, d_cand_count, d_ops);
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop);
            sec = ms / 1000.0;
            cudaMemcpy(&ops, d_ops, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            cudaMemcpy(&cand_cnt, d_cand_count, sizeof(int), cudaMemcpyDeviceToHost);
        }
        cudaEventDestroy(start); cudaEventDestroy(stop);
        std::cout << "ops=" << ops << " time(s)=" << sec << " cands/s=" << (sec > 0 ? (unsigned long long)(ops / sec) : 0ULL)
                  << " verifier-hits=" << cand_cnt << std::endl;
        if (cand_cnt >= AES_CAND_CAP) {
            std::cout << "WARNING: candidate slots overflowed (" << AES_CAND_CAP
                      << "); 16-bit verifier filter too weak for this space - use a multi-entry zip" << std::endl;
        }
        if (cand_cnt > 0) {
            std::vector<char> cands((size_t)cand_cnt * 96);
            std::vector<int> clens(cand_cnt);
            cudaMemcpy(cands.data(), d_cands, (size_t)cand_cnt * 96, cudaMemcpyDeviceToHost);
            cudaMemcpy(clens.data(), d_cand_lens, (size_t)cand_cnt * sizeof(int), cudaMemcpyDeviceToHost);
            for (int i = 0; i < cand_cnt; ++i) {
                std::string pwd(cands.data() + (size_t)i * 96, (size_t)clens[i]);
                if (verify_pwd(pwd)) {
                    std::cout << "FOUND pwd='" << pwd << "' (verified: PBKDF2 + HMAC-SHA1)" << std::endl;
                    return 0;
                }
            }
            std::cout << "all " << cand_cnt << " verifier hits were false positives" << std::endl;
        }
        std::cout << "NOT FOUND" << std::endl;
        return 0;
    }
    std::cerr << "usage:\n"
              << "  zipcrack --selftest\n"
              << "  zipcrack dict <zip> <dictfile>                       (ZipCrypto)\n"
              << "  zipcrack brute <zip> <preset|charset> <minL> <maxL>  (ZipCrypto; presets: lower/digits/lowerdigits/freq/mixed)\n"
              << "  zipcrack aes-make <out.zip> <pwd> <content>          (create AE-2 test zip)\n"
              << "  zipcrack aes-dict <zip> <dictfile>                   (WinZip AES)\n"
              << "  zipcrack aes-brute <zip> <preset|charset> <minL> <maxL>\n"
              << "  zipcrack aes-verify <zip> <pwd> [outdir]" << std::endl;
    return 1;
}
