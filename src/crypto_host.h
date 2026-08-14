// host-side crypto primitives: SHA-1, HMAC-SHA1, PBKDF2-HMAC-SHA1, AES (CTR)
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

static inline uint32_t rol32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

// ---------------- SHA-1 ----------------
struct Sha1Ctx { uint32_t h[5]; uint64_t total; unsigned char buf[64]; size_t buflen; };

static inline void sha1_init(Sha1Ctx& c) {
    c.h[0] = 0x67452301; c.h[1] = 0xefcdab89; c.h[2] = 0x98badcfe; c.h[3] = 0x10325476; c.h[4] = 0xc3d2e1f0;
    c.total = 0; c.buflen = 0;
}
static inline void sha1_block(uint32_t h[5], const unsigned char* p) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i)
        w[i] = ((uint32_t)p[i*4] << 24) | ((uint32_t)p[i*4+1] << 16) | ((uint32_t)p[i*4+2] << 8) | p[i*4+3];
    for (int i = 16; i < 80; ++i) w[i] = rol32(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;
        if (i < 20) { f = (b & c) | ((~b) & d); k = 0x5a827999u; }
        else if (i < 40) { f = b ^ c ^ d; k = 0x6ed9eba1u; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdcu; }
        else { f = b ^ c ^ d; k = 0xca62c1d6u; }
        uint32_t tmp = rol32(a, 5) + f + e + k + w[i];
        e = d; d = c; c = rol32(b, 30); b = a; a = tmp;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
}
static inline void sha1_update(Sha1Ctx& c, const unsigned char* data, size_t len) {
    c.total += len;
    while (len) {
        size_t take = 64 - c.buflen; if (take > len) take = len;
        memcpy(c.buf + c.buflen, data, take);
        c.buflen += take; data += take; len -= take;
        if (c.buflen == 64) { sha1_block(c.h, c.buf); c.buflen = 0; }
    }
}
static inline void sha1_final(Sha1Ctx& c, unsigned char out[20]) {
    uint64_t bits = c.total * 8;
    unsigned char pad = 0x80;
    sha1_update(c, &pad, 1);
    unsigned char zero = 0;
    while (c.buflen != 56) sha1_update(c, &zero, 1);
    unsigned char lenb[8];
    for (int i = 0; i < 8; ++i) lenb[i] = (unsigned char)(bits >> (56 - 8 * i));
    sha1_update(c, lenb, 8);
    for (int i = 0; i < 5; ++i) {
        out[i*4] = (unsigned char)(c.h[i] >> 24); out[i*4+1] = (unsigned char)(c.h[i] >> 16);
        out[i*4+2] = (unsigned char)(c.h[i] >> 8); out[i*4+3] = (unsigned char)c.h[i];
    }
}
static inline void sha1(const unsigned char* data, size_t len, unsigned char out[20]) {
    Sha1Ctx c; sha1_init(c); sha1_update(c, data, len); sha1_final(c, out);
}

// ---------------- HMAC-SHA1 ----------------
static inline void hmac_sha1(const unsigned char* key, size_t keylen,
                             const unsigned char* data, size_t datalen, unsigned char out[20]) {
    unsigned char k[64];
    memset(k, 0, 64);
    if (keylen > 64) sha1(key, keylen, k);
    else memcpy(k, key, keylen);
    unsigned char ipad[64], opad[64];
    for (int i = 0; i < 64; ++i) { ipad[i] = k[i] ^ 0x36; opad[i] = k[i] ^ 0x5c; }
    Sha1Ctx c; unsigned char tmp[20];
    sha1_init(c); sha1_update(c, ipad, 64); sha1_update(c, data, datalen); sha1_final(c, tmp);
    sha1_init(c); sha1_update(c, opad, 64); sha1_update(c, tmp, 20); sha1_final(c, out);
}

// ---------------- PBKDF2-HMAC-SHA1 ----------------
static inline void pbkdf2_hmac_sha1(const unsigned char* pw, size_t pwlen,
                                    const unsigned char* salt, size_t saltlen, int iters,
                                    unsigned char* out, size_t dklen) {
    size_t blocks = (dklen + 19) / 20;
    unsigned char u[20], t[20];
    std::vector<unsigned char> msg(saltlen + 4);
    memcpy(msg.data(), salt, saltlen);
    for (size_t b = 1; b <= blocks; ++b) {
        msg[saltlen] = (unsigned char)(b >> 24); msg[saltlen+1] = (unsigned char)(b >> 16);
        msg[saltlen+2] = (unsigned char)(b >> 8); msg[saltlen+3] = (unsigned char)b;
        hmac_sha1(pw, pwlen, msg.data(), msg.size(), u);
        memcpy(t, u, 20);
        for (int i = 1; i < iters; ++i) {
            hmac_sha1(pw, pwlen, u, 20, u);
            for (int j = 0; j < 20; ++j) t[j] ^= u[j];
        }
        size_t n = dklen - (b - 1) * 20;
        if (n > 20) n = 20;
        memcpy(out + (b - 1) * 20, t, n);
    }
}

// ---------------- AES (encrypt, CTR) ----------------
static const unsigned char AES_SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};
static const unsigned char AES_RCON[11] = { 0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36 };

struct AesKey { unsigned char rk[240]; int rounds; };

static inline void aes_key_expand(AesKey& k, const unsigned char* key, int keybits) {
    int nk = keybits / 32;          // 4/6/8
    k.rounds = nk + 6;              // 10/12/14
    for (int i = 0; i < nk * 4; ++i) k.rk[i] = key[i];
    for (int i = nk; i < 4 * (k.rounds + 1); ++i) {
        unsigned char t[4];
        t[0] = k.rk[(i-1)*4]; t[1] = k.rk[(i-1)*4+1]; t[2] = k.rk[(i-1)*4+2]; t[3] = k.rk[(i-1)*4+3];
        if (i % nk == 0) {
            unsigned char x = t[0];
            t[0] = AES_SBOX[t[1]] ^ AES_RCON[i / nk]; t[1] = AES_SBOX[t[2]]; t[2] = AES_SBOX[t[3]]; t[3] = AES_SBOX[x];
        } else if (nk > 6 && i % nk == 4) {
            t[0] = AES_SBOX[t[0]]; t[1] = AES_SBOX[t[1]]; t[2] = AES_SBOX[t[2]]; t[3] = AES_SBOX[t[3]];
        }
        k.rk[i*4]   = k.rk[(i-nk)*4]   ^ t[0];
        k.rk[i*4+1] = k.rk[(i-nk)*4+1] ^ t[1];
        k.rk[i*4+2] = k.rk[(i-nk)*4+2] ^ t[2];
        k.rk[i*4+3] = k.rk[(i-nk)*4+3] ^ t[3];
    }
}
static inline unsigned char xtime(unsigned char a) { return (unsigned char)((a << 1) ^ ((a & 0x80) ? 0x1b : 0x00)); }
static inline void aes_encrypt_block(const AesKey& k, const unsigned char in[16], unsigned char out[16]) {
    unsigned char s[16];
    for (int i = 0; i < 16; ++i) s[i] = in[i] ^ k.rk[i];
    for (int r = 1; r <= k.rounds; ++r) {
        unsigned char t[16];
        for (int i = 0; i < 16; ++i) t[i] = AES_SBOX[s[i]];
        // shift rows
        unsigned char u[16];
        u[0]=t[0]; u[1]=t[5]; u[2]=t[10]; u[3]=t[15];
        u[4]=t[4]; u[5]=t[9]; u[6]=t[14]; u[7]=t[3];
        u[8]=t[8]; u[9]=t[13]; u[10]=t[2]; u[11]=t[7];
        u[12]=t[12]; u[13]=t[1]; u[14]=t[6]; u[15]=t[11];
        if (r == k.rounds) {
            for (int i = 0; i < 16; ++i) out[i] = u[i] ^ k.rk[r*16 + i];
        } else {
            // mix columns (GF(2^8), x*2 = xtime, x*3 = xtime(x) ^ x)
            for (int c = 0; c < 4; ++c) {
                int i = c * 4;
                unsigned char a0=u[i], a1=u[i+1], a2=u[i+2], a3=u[i+3];
                s[i]   = xtime(a0) ^ (xtime(a1) ^ a1) ^ a2 ^ a3;
                s[i+1] = a0 ^ xtime(a1) ^ (xtime(a2) ^ a2) ^ a3;
                s[i+2] = a0 ^ a1 ^ xtime(a2) ^ (xtime(a3) ^ a3);
                s[i+3] = (xtime(a0) ^ a0) ^ a1 ^ a2 ^ xtime(a3);
            }
            for (int i = 0; i < 16; ++i) s[i] ^= k.rk[r*16 + i];   // AddRoundKey
        }
    }
}
static inline void aes_ctr_crypt(const AesKey& k, unsigned char* data, size_t len, unsigned char counter[16]) {
    size_t off = 0;
    while (off < len) {
        unsigned char keystream[16];
        aes_encrypt_block(k, counter, keystream);
        size_t n = len - off; if (n > 16) n = 16;
        for (size_t i = 0; i < n; ++i) data[off + i] ^= keystream[i];
        for (int i = 15; i >= 0; --i) { if (++counter[i] != 0) break; }
        off += n;
    }
}
