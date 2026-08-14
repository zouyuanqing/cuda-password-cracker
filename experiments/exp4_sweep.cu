// Experiment 4: block/thread configuration sweep for the specialized fixed-L=7 kernel.
// Usage: exp4_sweep.exe <duration_sec>
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

double bench(int threads, int blocks, int N, unsigned long long goal, unsigned long long* di, unsigned long long* ds) {
    unsigned long long z = 0; cudaMemcpy(di, &z, 8, cudaMemcpyHostToDevice); cudaMemcpy(ds, &z, 8, cudaMemcpyHostToDevice);
    auto t0 = std::chrono::high_resolution_clock::now();
    k_carry_fixed<7><<<blocks, threads>>>(N, goal, di, ds);
    cudaError_t err = cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    if (err != cudaSuccess) { std::cout << "  [launch error " << err << "]" << std::endl; return 0.0; }
    double sec = std::chrono::duration<double>(t1 - t0).count();
    unsigned long long n = 0; cudaMemcpy(&n, di, 8, cudaMemcpyDeviceToHost);
    return (double)n / sec;
}

int main(int argc, char** argv) {
    int duration = argc >= 2 ? std::stoi(argv[1]) : 1;
    int N = 26;
    std::string alpha = "abcdefghijklmnopqrstuvwxyz";
    char ha[64]; for (size_t i = 0; i < alpha.size(); ++i) ha[i] = alpha[i];
    cudaMemcpyToSymbol(d_alphabet, ha, 64);
    cudaMemcpyToSymbol(d_K, K64, sizeof(K64));
    int clock_khz = 0; cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    unsigned long long goal = (unsigned long long)clock_khz * 1000ULL * (unsigned long long)duration;
    int sm = 0; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    unsigned long long* di; cudaMalloc(&di, 8);
    unsigned long long* ds; cudaMalloc(&ds, 8);

    int mb_auto = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb_auto, k_carry_fixed<7>, 256, 0);
    std::cout << "== exp4 sweep N=" << N << " SM=" << sm << " occupancy(256thr)=" << mb_auto << " blk/SM" << std::endl;

    int thread_list[] = { 64, 128, 256, 512 };
    int bpsm_list[] = { 1, 2, 4, 6, 8, 12, 16, 24 };
    for (int ti = 0; ti < 4; ++ti) {
        int threads = thread_list[ti];
        for (int bi = 0; bi < 8; ++bi) {
            int bpsm = bpsm_list[bi];
            int blocks = bpsm * sm;
            // skip configs that exceed the hardware thread limit
            if ((long long)threads * bpsm > 2048) continue;
            double r = bench(threads, blocks, N, goal, di, ds);
            std::cout << "threads=" << threads << " blocks/SM=" << bpsm << " (" << blocks << " blocks): "
                      << (unsigned long long)r << " H/s (" << r/1e9 << " GH/s)" << std::endl;
        }
    }
    cudaFree(di); cudaFree(ds);
    return 0;
}
