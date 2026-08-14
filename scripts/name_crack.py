# -*- coding: utf-8 -*-
# Brute-force WinZip AES (AE-2) candidates derived from the jpg filenames.
# Verifies: PBKDF2 verifier (2 bytes) + AE-2 HMAC-SHA1 over verifier||ciphertext.
import hashlib, hmac, struct, sys, itertools

ZIP = sys.argv[1] if len(sys.argv) > 1 else 'target.zip'
data = open(ZIP, 'rb').read()

def parse_entries():
    entries = []
    p = 0
    while p + 30 <= len(data):
        if data[p:p+4] == b'PK\x03\x04':
            flags = struct.unpack('<H', data[p+6:p+8])[0]
            method = struct.unpack('<H', data[p+8:p+10])[0]
            name_len = struct.unpack('<H', data[p+26:p+28])[0]
            extra_len = struct.unpack('<H', data[p+28:p+30])[0]
            csize = struct.unpack('<I', data[p+18:p+22])[0]
            if method == 99 and (flags & 1):
                x = p + 30 + name_len
                xe = x + extra_len
                strength = 3
                while x + 4 <= xe:
                    eid, esz = struct.unpack('<HH', data[x:x+4])
                    if eid == 0x9901:
                        strength = data[x+8]
                    x += 4 + esz
                saltlen = 16 if strength == 3 else 8
                s = p + 30 + name_len + extra_len
                salt = data[s:s+saltlen]
                ver = data[s+saltlen:s+saltlen+2]
                doff = s + saltlen + 2
                entries.append((salt, ver, doff, csize))
            p = p + 30 + name_len + extra_len + csize
            continue
        if data[p:p+4] == b'PK\x01\x02' or data[p:p+4] == b'PK\x05\x06':
            break
        p += 1
    return entries

def check(pw_bytes):
    for salt, ver, doff, csize in entries:
        dk = hashlib.pbkdf2_hmac('sha1', pw_bytes, salt, 1000, 34)
        if dk[32:34] != ver:
            return False
        h = hmac.new(dk[:32], ver + data[doff:doff+csize], hashlib.sha1).digest()[:10]
        if h != data[doff+csize:doff+csize+10]:
            return False
    return True

entries = parse_entries()
print(f'parsed {len(entries)} AES entries')

cands = []
# pinyin (encoded as UTF-8, the standard WinZip AES password encoding)
for w in ['xiaosanyue', 'changyeyue', 'xiaosan', 'changye', 'sanyue', 'yue']:
    for v in [w, w.capitalize(), w.upper()]:
        cands.append(v.encode('utf-8'))
# abbreviations
for w in ['xsy', 'cyy', 'xs', 'cy', 'syy', 'cyyue', 'xsyue']:
    for v in [w, w.capitalize(), w.upper()]:
        cands.append(v.encode('utf-8'))
# pinyin + numbers
for w in ['xiaosanyue', 'changyeyue', 'xsy', 'cyy', 'xiaosan', 'changye']:
    for suf in ['', '1', '12', '123', '2024', '2025', '2026', '520', '1314', '666', '888']:
        cands.append((w + suf).encode('utf-8'))
        cands.append((suf + w).encode('utf-8'))
        cands.append((w.capitalize() + suf).encode('utf-8'))
# Chinese (UTF-8 and GBK encodings)
for zh in ['小三月', '长夜月', '三月', '长夜', '小三月2025', '长夜月2025', '三月2025']:
    cands.append(zh.encode('utf-8'))
    cands.append(zh.encode('gbk'))
    cands.append(zh.encode('gbk') + b'2025')
    cands.append(b'2025' + zh.encode('gbk'))

seen = set()
hit = None
for c in cands:
    if c in seen:
        continue
    seen.add(c)
    if check(c):
        hit = c
        break

if hit:
    print('FOUND password bytes =', hit)
    try:
        print('UTF-8 decode:', hit.decode('utf-8'))
    except Exception:
        pass
    try:
        print('GBK decode:', hit.decode('gbk'))
    except Exception:
        pass
else:
    print(f'no hit in {len(seen)} candidates')
