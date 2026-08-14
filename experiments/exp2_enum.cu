// Experiment 2: enumeration strategy — base-N div/mod vs carry-increment —
// and hash-path specialization (runtime-L sha256_device vs compile-time-L sha256_fixed).
// Usage: exp2_enum.exe <duration_sec> <N>
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <string>
#include <chrono>
#include "sha256_common.cuh"

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

// carry-increment, runtime-L hash path (sha256_device)
template<int L>
__global__ void k_carry_device(int N, unsigned long long goal, unsigned long long* iters, unsigned long long* sink) {
    unsigned long long start = clock64();
    uint64_t base = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    unsigned int d[L];
    uint64_t t = base;
#pragma unroll
    for (int i = 0; i < L; ++i) { d[i] = (unsigned int)(t % (uint64_t)N); t /= (uint64_t)N; }
    char buf[L];
    uint32_t acc = 0; unsigned long long it = 0;
    while (clock64() - start < goal) {
#pragma unroll
        for (int i = 0; i < L; ++i) buf[i] = d_alphabet[d[i]];
        uint32_t h[8];
        sha256_device(buf, L, h);
        acc += h[0];
        unsigned int carry = 1;
#pragma unroll
        for (int i = 0; i < L; ++i) {
            unsigned int s = d[i] + carry;
            if (s >= (unsigned int)N) { s -= (unsigned int)N; carry = 1; } else carry = 0;
            d[i] = s;
        }
        ++it;
    }
    atomicAdd(iters, it);
    if (acc == 0xDEADBEEFu) atomicAdd(sink, 1ULL);
}

// carry-increment, compile-time-L hash path (sha256_fixed)
template<int L>
__global__ void k_carry_fixed(int N, unsigned long long goal, unsigned long long* iters, unsigned long long* sink) {
    unsigned long long start = clock64();
    uint64_t base = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    unsigned int d[L];
    uint64_t t = base;
#pragma unroll
    for (int i = 0; i < L; ++i) { d[i] = (unsigned int)(t % (uint64_t)N); t /= (uint64_t)N; }
    char buf[L];
    uint32_t acc = 0; unsigned long long it = 0;
    while (clock64() - start < goal) {
#pragma unroll
        for (int i = 0; i < L; ++i) buf[i] = d_alphabet[d[i]];
        uint32_t h[8];
        sha256_fixed<L>(buf, h);
        acc += h[0];
        unsigned int carry = 1;
#pragma unroll
        for (int i = 0; i < L; ++i) {
            unsigned int s = d[i] + carry;
            if (s >= (unsigned int)N) { s -= (unsigned int)N; carry = 1; } else carry = 0;
            d[i] = s;
        }
        ++it;
    }
    atomicAdd(iters, it);
    if (acc == 0xDEADBEEFu) atomicAdd(sink, 1ULL);
}

// div/mod enumeration, runtime-L hash path
template<int L>
__global__ void k_divmod_device(int N, unsigned long long goal, unsigned long long* iters, unsigned long long* sink) {
    unsigned long long start = clock64();
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    char buf[L];
    uint32_t acc = 0; unsigned long long it = 0;
    while (clock64() - start < goal) {
        uint64_t x = idx;
#pragma unroll
        for (int i = 0; i < L; ++i) { buf[i] = d_alphabet[x % (uint64_t)N]; x /= (uint64_t)N; }
        uint32_t h[8];
        sha256_device(buf, L, h);
        acc += h[0];
        idx += stride;
        ++it;
    }
    atomicAdd(iters, it);
    if (acc == 0xDEADBEEFu) atomicAdd(sink, 1ULL);
}

template<int L>
void bench_all(int N, unsigned long long goal, int sm) {
    int threads = 256;
    unsigned long long* di; cudaMalloc(&di, 8);
    unsigned long long* ds; cudaMalloc(&ds, 8);
    unsigned long long z = 0;
    int mb; int blocks;
    double sec; unsigned long long n;

    // carry + device
    mb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k_carry_device<L>, threads, 0); blocks = mb * sm;
    cudaMemcpy(di, &z, 8, cudaMemcpyHostToDevice); cudaMemcpy(ds, &z, 8, cudaMemcpyHostToDevice);
    { auto t0 = std::chrono::high_resolution_clock::now(); k_carry_device<L><<<blocks,threads>>>(N, goal, di, ds); cudaDeviceSynchronize(); auto t1 = std::chrono::high_resolution_clock::now(); sec = std::chrono::duration<double>(t1-t0).count(); }
    cudaMemcpy(&n, di, 8, cudaMemcpyDeviceToHost);
    std::cout << "L=" << L << " carry+device : " << (unsigned long long)(n/sec) << " H/s (" << n/sec/1e9 << " GH/s) blocks=" << blocks << " mbpsm=" << mb << std::endl;

    // carry + fixed
    mb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k_carry_fixed<L>, threads, 0); blocks = mb * sm;
    cudaMemcpy(di, &z, 8, cudaMemcpyHostToDevice); cudaMemcpy(ds, &z, 8, cudaMemcpyHostToDevice);
    { auto t0 = std::chrono::high_resolution_clock::now(); k_carry_fixed<L><<<blocks,threads>>>(N, goal, di, ds); cudaDeviceSynchronize(); auto t1 = std::chrono::high_resolution_clock::now(); sec = std::chrono::duration<double>(t1-t0).count(); }
    cudaMemcpy(&n, di, 8, cudaMemcpyDeviceToHost);
    std::cout << "L=" << L << " carry+fixed  : " << (unsigned long long)(n/sec) << " H/s (" << n/sec/1e9 << " GH/s) blocks=" << blocks << " mbpsm=" << mb << std::endl;

    // divmod + device
    mb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k_divmod_device<L>, threads, 0); blocks = mb * sm;
    cudaMemcpy(di, &z, 8, cudaMemcpyHostToDevice); cudaMemcpy(ds, &z, 8, cudaMemcpyHostToDevice);
    { auto t0 = std::chrono::high_resolution_clock::now(); k_divmod_device<L><<<blocks,threads>>>(N, goal, di, ds); cudaDeviceSynchronize(); auto t1 = std::chrono::high_resolution_clock::now(); sec = std::chrono::duration<double>(t1-t0).count(); }
    cudaMemcpy(&n, di, 8, cudaMemcpyDeviceToHost);
    std::cout << "L=" << L << " divmod+device: " << (unsigned long long)(n/sec) << " H/s (" << n/sec/1e9 << " GH/s) blocks=" << blocks << " mbpsm=" << mb << std::endl;

    cudaFree(di); cudaFree(ds);
}

int main(int argc, char** argv) {
    int duration = argc >= 2 ? std::stoi(argv[1]) : 3;
    int N = argc >= 3 ? std::stoi(argv[2]) : 26;
    std::string alpha = "abcdefghijklmnopqrstuvwxyz";
    char ha[64]; for (size_t i = 0; i < alpha.size() && i < 64; ++i) ha[i] = alpha[i];
    cudaMemcpyToSymbol(d_alphabet, ha, 64);
    cudaMemcpyToSymbol(d_K, K64, sizeof(K64));
    int clock_khz = 0; cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    unsigned long long goal = (unsigned long long)clock_khz * 1000ULL * (unsigned long long)duration;
    int sm = 0; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    std::cout << "== exp2 enumeration N=" << N << " goal_cycles=" << goal << " SM=" << sm << std::endl;

    bench_all<5>(N, goal, sm);
    bench_all<6>(N, goal, sm);
    bench_all<7>(N, goal, sm);
    bench_all<8>(N, goal, sm);
    return 0;
}
