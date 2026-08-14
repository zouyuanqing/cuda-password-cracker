// Experiment 5: dictionary scan memory layout — baseline SoA (flat+offsets+lens)
// vs padded fixed-width (64B slots, uint4 coalesced loads).
// Usage: exp5_dict.exe <dict_path>
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
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

// baseline: SoA flat + offsets + lens (mirrors dict_kernel)
__global__ void k_soa(const char* flat, const int* offsets, const int* lens, int count, unsigned long long* ops) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    uint32_t acc = 0;
    while (idx < (uint64_t)count) {
        const char* msg = flat + offsets[(int)idx];
        int len = lens[(int)idx];
        uint32_t h[8];
        if (len <= 55) sha256_device(msg, len, h); else sha256_device(msg, 55, h); // avoid multi for bench (rockyou len<=1024)
        acc += h[0];
        idx += stride;
    }
    if (acc == 0xDEADBEEFu) atomicAdd(ops, 1ULL);
}

// padded fixed-width: each entry occupies 64 bytes (16 uint32), coalesced uint4 loads
__global__ void k_padded(const uint32_t* __restrict__ padded, const int* __restrict__ lens, int count, unsigned long long* ops) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    uint32_t acc = 0;
    while (idx < (uint64_t)count) {
        unsigned char e[64];
        const uint4* src = (const uint4*)(padded + idx * 16);
        ((uint4*)e)[0] = src[0];
        ((uint4*)e)[1] = src[1];
        ((uint4*)e)[2] = src[2];
        ((uint4*)e)[3] = src[3];
        int len = lens[idx];
        uint32_t h[8];
        if (len <= 55) sha256_device((const char*)e, len, h); else sha256_device((const char*)e, 55, h);
        acc += h[0];
        idx += stride;
    }
    if (acc == 0xDEADBEEFu) atomicAdd(ops, 1ULL);
}

double run(void (*k)(const char*,const int*,const int*,int,unsigned long long*),
           const char* flat, const int* offs, const int* lens, int count, int threads, unsigned long long* ops) {
    int blocks = (count + threads - 1) / threads; if (blocks > 65535) blocks = 65535;
    unsigned long long z = 0; cudaMemcpy(ops, &z, 8, cudaMemcpyHostToDevice);
    auto t0 = std::chrono::high_resolution_clock::now();
    k<<<blocks, threads>>>(flat, offs, lens, count, ops);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

int main(int argc, char** argv) {
    std::string path = argc >= 2 ? std::string(argv[1]) : "candidates.txt";
    cudaMemcpyToSymbol(d_K, K64, sizeof(K64));
    const int MAX_PWD = 1024;
    std::vector<int> offsets, lens; std::vector<char> flat;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if ((int)line.size() > MAX_PWD) continue;
        offsets.push_back((int)flat.size());
        lens.push_back((int)line.size());
        flat.insert(flat.end(), line.begin(), line.end());
    }
    int count = (int)offsets.size();
    std::cout << "exp5 dict loaded lines=" << count << " bytes=" << flat.size() << std::endl;

    // device buffers: SoA
    char* d_flat; cudaMalloc(&d_flat, flat.size()); cudaMemcpy(d_flat, flat.data(), flat.size(), cudaMemcpyHostToDevice);
    int* d_offs; cudaMalloc(&d_offs, count*4); cudaMemcpy(d_offs, offsets.data(), count*4, cudaMemcpyHostToDevice);
    int* d_lens; cudaMalloc(&d_lens, count*4); cudaMemcpy(d_lens, lens.data(), count*4, cudaMemcpyHostToDevice);
    // device buffers: padded (64B per entry)
    std::vector<uint32_t> padded((size_t)count * 16, 0u);
    for (int i = 0; i < count; ++i) {
        const char* src = flat.data() + offsets[i];
        int l = lens[i];
        unsigned char* dst = (unsigned char*)(padded.data() + (size_t)i * 16);
        for (int j = 0; j < l; ++j) dst[j] = src[j];
    }
    uint32_t* d_padded; cudaMalloc(&d_padded, (size_t)count * 16 * 4);
    cudaMemcpy(d_padded, padded.data(), (size_t)count * 16 * 4, cudaMemcpyHostToDevice);
    unsigned long long* d_ops; cudaMalloc(&d_ops, 8);

    for (int threads : {128, 256}) {
        double t_soa = run((void(*)(const char*,const int*,const int*,int,unsigned long long*))k_soa, d_flat, d_offs, d_lens, count, threads, d_ops);
        std::cout << "SOA     threads=" << threads << ": " << t_soa*1000.0 << " ms  (" << (double)count/t_soa/1e9 << " GH/s)" << std::endl;
    }
    {
        int threads = 256;
        int blocks = (count + threads - 1)/threads; if (blocks > 65535) blocks = 65535;
        unsigned long long z=0; cudaMemcpy(d_ops,&z,8,cudaMemcpyHostToDevice);
        auto t0 = std::chrono::high_resolution_clock::now();
        k_padded<<<blocks,threads>>>(d_padded, d_lens, count, d_ops);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double>(t1-t0).count();
        std::cout << "PADDED  threads=256: " << t*1000.0 << " ms  (" << (double)count/t/1e9 << " GH/s)" << std::endl;
    }

    cudaFree(d_flat); cudaFree(d_offs); cudaFree(d_lens); cudaFree(d_padded); cudaFree(d_ops);
    return 0;
}
