// Experiment 6: PCIe upload cost — pageable vs pinned (cudaHostAlloc) for the
// ~240MB dict transfer (125MB flat + 57MB offsets + 57MB lens), and async+stream overlap.
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <chrono>

static double bench_memcpy(size_t bytes, void* hsrc, void* dsrc, bool pinned) {
    // warmup
    cudaMemcpy(dsrc, hsrc, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i) cudaMemcpy(dsrc, hsrc, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count() / 5.0;
    return bytes / sec / 1e9; // GB/s
}

int main() {
    size_t total = 240 * 1024 * 1024; // flat 125 + offsets 57 + lens 57 ~= 240MB
    std::vector<char> hbuf(total, 0x5a);
    char* d; cudaMalloc(&d, total);

    double gbps_pageable = bench_memcpy(total, hbuf.data(), d, false);

    char* hp; cudaHostAlloc(&hp, total, cudaHostAllocDefault);
    for (size_t i = 0; i < total; ++i) hp[i] = 0x5a;
    double gbps_pinned = bench_memcpy(total, hp, d, true);

    std::cout << "240MB H2D pageable: " << gbps_pageable << " GB/s (" << (total/ (gbps_pageable*1e9))*1e3 << " ms)" << std::endl;
    std::cout << "240MB H2D pinned  : " << gbps_pinned << " GB/s (" << (total/ (gbps_pinned*1e9))*1e3 << " ms)" << std::endl;

    // single 125MB flat (the largest chunk)
    size_t flat = 125 * 1024 * 1024;
    double gb_p = bench_memcpy(flat, hbuf.data(), d, false);
    double gb_pn = bench_memcpy(flat, hp, d, true);
    std::cout << "125MB H2D pageable: " << gb_p << " GB/s (" << (flat/(gb_p*1e9))*1e3 << " ms)" << std::endl;
    std::cout << "125MB H2D pinned  : " << gb_pn << " GB/s (" << (flat/(gb_pn*1e9))*1e3 << " ms)" << std::endl;

    cudaFree(d); cudaFreeHost(hp);
    return 0;
}
