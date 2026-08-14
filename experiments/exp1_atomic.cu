// Experiment 1: quantify the per-hash global atomicAdd cost, and verify whether
// the stress kernel's hash result is dead-code-eliminated (h is never read there).
// Usage: exp1_atomic.exe <duration_sec> <L> <alphabet_size> <threads>
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

// A: faithful copy of stress_kernel (h result NOT used -> hash is DCE-able)
__global__ void k_orig_stress(int L, int N, unsigned long long goal, unsigned long long* ops) {
    unsigned long long start = clock64();
    char buf[32];
    unsigned long long s = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    while (clock64() - start < goal) {
        s += 1ULL;
        for (int i = 0; i < L; ++i) buf[i] = d_alphabet[(s + (unsigned long long)i) % (unsigned long long)N];
        uint32_t h[8];
        sha256_device(buf, L, h);
        atomicAdd(ops, 1ULL);
    }
}

// B: hash result sunk into per-thread accumulator, NO per-iter atomic
__global__ void k_hash_sink(int L, int N, unsigned long long goal, unsigned long long* iters, unsigned long long* sink) {
    unsigned long long start = clock64();
    char buf[32];
    unsigned long long s = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long it = 0;
    uint32_t acc = 0;
    while (clock64() - start < goal) {
        s += 1ULL;
        for (int i = 0; i < L; ++i) buf[i] = d_alphabet[(s + (unsigned long long)i) % (unsigned long long)N];
        uint32_t h[8];
        sha256_device(buf, L, h);
        acc += h[0];
        ++it;
    }
    atomicAdd(iters, it);
    if (acc == 0xDEADBEEFu) atomicAdd(sink, 1ULL);
}

// C: hash sink + global atomicAdd per hash (real brute/dict behaviour)
__global__ void k_hash_atomic(int L, int N, unsigned long long goal, unsigned long long* ops, unsigned long long* sink) {
    unsigned long long start = clock64();
    char buf[32];
    unsigned long long s = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    while (clock64() - start < goal) {
        s += 1ULL;
        for (int i = 0; i < L; ++i) buf[i] = d_alphabet[(s + (unsigned long long)i) % (unsigned long long)N];
        uint32_t h[8];
        sha256_device(buf, L, h);
        acc += h[0];
        atomicAdd(ops, 1ULL);
    }
    if (acc == 0xDEADBEEFu) atomicAdd(sink, 1ULL);
}

// D: hash sink + shared-memory atomicAdd per hash, one global atomic per block at end
__global__ void k_hash_block(int L, int N, unsigned long long goal, unsigned long long* ops, unsigned long long* sink) {
    __shared__ unsigned long long s_cnt;
    unsigned long long start = clock64();
    char buf[32];
    unsigned long long s = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    if (threadIdx.x == 0) s_cnt = 0;
    __syncthreads();
    while (clock64() - start < goal) {
        s += 1ULL;
        for (int i = 0; i < L; ++i) buf[i] = d_alphabet[(s + (unsigned long long)i) % (unsigned long long)N];
        uint32_t h[8];
        sha256_device(buf, L, h);
        acc += h[0];
        atomicAdd(&s_cnt, 1ULL);
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(ops, s_cnt);
    if (acc == 0xDEADBEEFu) atomicAdd(sink, 1ULL);
}

static double run_kernel(void (*k)(int,int,unsigned long long,unsigned long long*,unsigned long long*),
                         int L, int N, unsigned long long goal, int threads, int blocks,
                         unsigned long long* d_count, unsigned long long* d_sink) {
    unsigned long long zero = 0; cudaMemcpy(d_count, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sink, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    auto t0 = std::chrono::high_resolution_clock::now();
    k<<<blocks, threads>>>(L, N, goal, d_count, d_sink);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    unsigned long long n = 0; cudaMemcpy(&n, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    return (double)n / sec;
}

int main(int argc, char** argv) {
    int duration = argc >= 2 ? std::stoi(argv[1]) : 3;
    int L = argc >= 3 ? std::stoi(argv[2]) : 7;
    std::string alpha = argc >= 4 ? std::string(argv[3]) : std::string("abcdefghijklmnopqrstuvwxyz");
    int threads = argc >= 5 ? std::stoi(argv[4]) : 256;
    int N = (int)alpha.size();
    char ha[64]; for (size_t i = 0; i < alpha.size(); ++i) ha[i] = alpha[i];
    cudaMemcpyToSymbol(d_alphabet, ha, 64);
    cudaMemcpyToSymbol(d_K, K64, sizeof(K64));
    int clock_khz = 0; cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    unsigned long long goal = (unsigned long long)clock_khz * 1000ULL * (unsigned long long)duration;
    int sm = 0; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);

    unsigned long long* d_count; cudaMalloc(&d_count, sizeof(unsigned long long));
    unsigned long long* d_sink;  cudaMalloc(&d_sink, sizeof(unsigned long long));

    std::cout << "== exp1 atomicAdd analysis: L=" << L << " N=" << N << " threads=" << threads
              << " goal_cycles=" << goal << " SM=" << sm << std::endl;

    // A: original stress (hash result DCE'd)
    {
        unsigned long long zero = 0; cudaMemcpy(d_count, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        auto t0 = std::chrono::high_resolution_clock::now();
        int mb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k_orig_stress, threads, 0);
        int blocks = mb * sm;
        k_orig_stress<<<blocks, threads>>>(L, N, goal, d_count);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        unsigned long long n = 0; cudaMemcpy(&n, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        std::cout << "A orig_stress (h unused, atomic/hash) : " << (unsigned long long)(n/sec) << " H/s  (" << n/sec/1e9 << " GH/s)" << std::endl;
    }
    // B: hash sink, no per-iter atomic
    {
        int mb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k_hash_sink, threads, 0);
        int blocks = mb * sm;
        double r = run_kernel(k_hash_sink, L, N, goal, threads, blocks, d_count, d_sink);
        std::cout << "B hash_sink (no per-iter atomic)        : " << (unsigned long long)r << " H/s  (" << r/1e9 << " GH/s)  blocks=" << blocks << " mbpsm=" << mb << std::endl;
    }
    // C: hash sink + global atomic per hash
    {
        int mb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k_hash_atomic, threads, 0);
        int blocks = mb * sm;
        double r = run_kernel(k_hash_atomic, L, N, goal, threads, blocks, d_count, d_sink);
        std::cout << "C hash+global atomic/hash               : " << (unsigned long long)r << " H/s  (" << r/1e9 << " GH/s)  blocks=" << blocks << " mbpsm=" << mb << std::endl;
    }
    // D: hash sink + shared atomic per hash
    {
        int mb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k_hash_block, threads, 0);
        int blocks = mb * sm;
        double r = run_kernel(k_hash_block, L, N, goal, threads, blocks, d_count, d_sink);
        std::cout << "D hash+shared atomic/hash               : " << (unsigned long long)r << " H/s  (" << r/1e9 << " GH/s)  blocks=" << blocks << " mbpsm=" << mb << std::endl;
    }

    cudaFree(d_count); cudaFree(d_sink);
    return 0;
}
