// ZipCrypto (PKZIP traditional encryption) cracker - CPU reference
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>

static uint32_t crc_table[256];

static void init_crc() {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
        crc_table[i] = c;
    }
}
#define CRC32(c, b) (crc_table[((c) ^ (b)) & 0xff] ^ ((c) >> 8))

static inline void update_keys(uint32_t& k0, uint32_t& k1, uint32_t& k2, unsigned char c) {
    k0 = CRC32(k0, c);
    k1 = (k1 + (k0 & 0xff)) * 134775813u + 1u;
    k2 = CRC32(k2, (unsigned char)(k1 >> 24));
}
static inline unsigned char ks_byte(uint32_t k2) {
    uint32_t t = k2 | 2;
    return (unsigned char)((t * (t ^ 1)) >> 8);
}

// decrypt the 12-byte verification header with candidate password, compare check byte
static bool check_pw(const char* pw, int len, const unsigned char* ver, unsigned char check) {
    uint32_t k0 = 0x12345678u, k1 = 0x23456789u, k2 = 0x34567890u;
    for (int i = 0; i < len; ++i) update_keys(k0, k1, k2, (unsigned char)pw[i]);
    for (int i = 0; i < 11; ++i) {
        unsigned char c = ver[i] ^ ks_byte(k2);
        update_keys(k0, k1, k2, c);
    }
    unsigned char c11 = ver[11] ^ ks_byte(k2);
    return c11 == check;
}

static const unsigned char VER[12] = { 0xc6,0x68,0x3b,0x97,0x4d,0x0d,0x29,0x80,0xb0,0x47,0xcc,0x5e };
static const unsigned char CHECK_BYTE = 0x6b;

// all six encrypted entries: 12-byte verification header + check byte (modtime>>8, bit3 set)
struct EntryVer { unsigned char ver[12]; unsigned char check; };
static const EntryVer ALL_ENTRIES[6] = {
    { { 0xc6,0x68,0x3b,0x97,0x4d,0x0d,0x29,0x80,0xb0,0x47,0xcc,0x5e }, 0x6b }, // bin1/11.30xjhh.smart
    { { 0xf6,0xf1,0x13,0x0a,0x4a,0x32,0x2d,0xee,0xdb,0xed,0xab,0x11 }, 0x89 }, // bin1/12.30xjh.emtp
    { { 0x72,0xdd,0xbb,0xc7,0x26,0xc3,0x20,0x19,0x02,0x23,0x7c,0x48 }, 0x68 }, // MainModule.mod
    { { 0x21,0x14,0x29,0xad,0xb9,0xfc,0x3d,0xb0,0xb0,0x78,0x99,0xc5 }, 0x68 }, // Program01.pgf
    { { 0x4e,0x9f,0x65,0x9f,0xa2,0x96,0xb2,0x1e,0x55,0xfd,0x3e,0x1c }, 0x68 }, // xs_exp_1.mod
    { { 0x63,0x41,0x82,0x40,0xd5,0x63,0x6b,0x9d,0x42,0x4c,0x88,0x2a }, 0x68 }, // xs_fj_exp1.mod
};

// stage-2: candidate must decrypt the verification header of ALL six entries
static bool check_pw_all(const char* pw, int len) {
    for (int e = 0; e < 6; ++e) {
        if (!check_pw(pw, len, ALL_ENTRIES[e].ver, ALL_ENTRIES[e].check)) return false;
    }
    return true;
}

static bool brute_rec(std::string& buf, int pos, int L, const std::string& charset, long long& n) {
    if (pos == L) {
        n++;
        if (check_pw(buf.c_str(), L, VER, CHECK_BYTE)) {
            return check_pw_all(buf.c_str(), L);
        }
        return false;
    }
    for (size_t i = 0; i < charset.size(); ++i) {
        buf[pos] = charset[i];
        if (brute_rec(buf, pos + 1, L, charset, n)) return true;
    }
    return false;
}

int main(int argc, char** argv) {
    init_crc();
    if (argc >= 3 && std::string(argv[1]) == "dict") {
        std::ifstream in(argv[2]);
        if (!in) { std::cerr << "open dict failed" << std::endl; return 1; }
        std::string line;
        long long n = 0;
        auto t0 = std::chrono::steady_clock::now();
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            n++;
            if (check_pw(line.c_str(), (int)line.size(), VER, CHECK_BYTE)) {
                // stage-2: verify against all six entries (48-bit filter, kills false positives)
                if (check_pw_all(line.c_str(), (int)line.size())) {
                    auto t1 = std::chrono::steady_clock::now();
                    double sec = std::chrono::duration<double>(t1 - t0).count();
                    std::cout << "FOUND pwd='" << line << "' line=" << n << " time(s)=" << sec
                              << " rate=" << (unsigned long long)(n / sec) << "/s" << std::endl;
                    return 0;
                }
            }
            if (n % 5000000 == 0) {
                auto t1 = std::chrono::steady_clock::now();
                double sec = std::chrono::duration<double>(t1 - t0).count();
                std::cout << "PROGRESS " << n << " rate=" << (unsigned long long)(n / sec) << "/s" << std::endl;
            }
        }
        std::cout << "not found in " << n << " lines" << std::endl;
        return 0;
    }
    if (argc >= 5 && std::string(argv[1]) == "brute") {
        std::string charset = argv[2];
        int minL = std::stoi(argv[3]);
        int maxL = std::stoi(argv[4]);
        long long n = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int L = minL; L <= maxL; ++L) {
            std::string buf(L, ' ');
            std::cout << "brute L=" << L << " space=" << (unsigned long long)std::pow((double)charset.size(), L) << std::endl;
            if (brute_rec(buf, 0, L, charset, n)) {
                std::cout << "FOUND pwd='" << buf << "'" << std::endl;
                return 0;
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "not found, tried=" << n << " time(s)=" << sec << " rate=" << (unsigned long long)(n / sec) << "/s" << std::endl;
        return 0;
    }
    std::cerr << "usage: zipcrack_cpu dict <file> | brute <charset> <minL> <maxL>" << std::endl;
    return 1;
}
