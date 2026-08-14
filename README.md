# CUDA Password Cracker

基于 CUDA 的密码破解工具包，包含两个独立工具：

| 工具 | 目标 | 实测性能（RTX 5070 Laptop, 36 SM） |
|------|------|-------------------------------------|
| `sha256` | 任意 SHA-256 哈希的字典/暴力破解 | 暴力 **4.8 GH/s**（≈88-94% hashcat per-SM）；字典 4.2 GH/s |
| `zipcrack` | ZIP 压缩包密码（ZipCrypto + WinZip AES AE-2） | ZipCrypto 暴力 **1.4-1.5 GH/s**；AES-256 暴力 **1.05M 候选/s**（hashcat 同量级 per-SM） |

全部实现为纯标量 CUDA 内核，无第三方依赖，已针对 Blackwell（sm_120）与 CUDA 13.2 调优。

## 功能总览

### sha256 —— SHA-256 破解器

- 完整多块 SHA-256（任意消息长度，CPU/GPU 双实现交叉验证）
- 字典模式：整库一次上传常驻显存，多目标复用，Markov 2 阶排序（`--no-markov` 关闭）
- 暴力模式：编译期定长 L 的进位增量枚举（1..32 位），`--progress` 分块进度
- 基准模式：`--stress` 实测真实哈希吞吐（已修复死代码消除假象）

### zipcrack —— ZIP 破解器

- **ZipCrypto**：CRC32 密钥调度内核，48 位多条目过滤（零假阳性），block-ticket 多长度并行扫描
- **WinZip AES (AE-2)**：设备端 PBKDF2-HMAC-SHA1(1000) 内核 + 双条目 32 位过滤 + 主机端 HMAC 终审（假阳性 2⁻³²）
- 完整密码学库（`crypto_host.h`）：SHA-1 / HMAC-SHA1 / PBKDF2 / AES-CTR，RFC 6070 + FIPS-197 测试向量全过（`--selftest`）
- 辅助能力：`aes-make` 生成 AE-2 测试包、`aes-verify` 验证密码并解密提取
- 字符集预设：`lower` / `digits` / `lowerdigits` / `freq`（Markov 频率序）/ `mixed`

## 环境要求

- Windows + NVIDIA GPU（计算能力 ≥ 7.0）
- CUDA Toolkit 12.x/13.x（本项目在 CUDA 13.2 上开发；代码兼容 12.x）
- MSVC 构建工具链（随 Visual Studio 安装）

## 构建

```powershell
# 方式一：PowerShell 脚本（推荐，Windows）
.\build.ps1            # 输出到 build\

# 方式二：CMake（需安装 CMake ≥ 3.18）
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release

# 方式三：手动 nvcc
nvcc -O3 -arch=sm_120 src\brute_sha256.cu -o build\brute_sha256.exe
nvcc -O3 -arch=sm_120 src\zipcrack.cu      -o build\zipcrack.exe
nvcc -O2 src\zipcrack_cpu.cpp              -o build\zipcrack_cpu.exe
```

> 运行 exe 前需保证 CUDA bin 目录在 PATH 中（如 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin`）。

## 用法

### SHA-256 破解（brute_sha256.exe）

```powershell
# 计算哈希（CPU 参照）
brute_sha256.exe --hash abc

# 暴力破解：长度 字母表 目标明文
brute_sha256.exe 5 abcdefghijklmnopqrstuvwxyz abcde
brute_sha256.exe 7 abcdefghijklmnopqrstuvwxyz ABCDEFG --progress   # 大空间 + 进度输出

# 字典攻击：字典文件 目标明文（可多目标，整库常驻显存）
brute_sha256.exe --dict candidates.txt password 123456
brute_sha256.exe --dict candidates.txt target --no-markov          # 关闭 Markov 排序

# 基准测试：真实 SHA-256 吞吐（秒数 长度 字母表）
brute_sha256.exe --stress 3 7 abcdefghijklmnopqrstuvwxyz
```

### ZIP 破解（zipcrack.exe）

```powershell
# 自检：PBKDF2/HMAC/AES 测试向量（host + device）
zipcrack.exe --selftest

# ZipCrypto 字典 / 暴力
zipcrack.exe dict  secret.zip rockyou.txt
zipcrack.exe brute secret.zip freq 1 7                # 频率序字符集，命中时间最优

# WinZip AES (AE-2)
zipcrack.exe aes-dict  secret.zip rockyou.txt
zipcrack.exe aes-brute secret.zip lowerdigits 1 6

# 验证密码 + 解密提取 / 生成测试包
zipcrack.exe aes-verify secret.zip <password> outdir
zipcrack.exe aes-make  test-aes.zip <password> "content"
```

字符集预设：`lower`（a-z）、`digits`（0-9）、`lowerdigits`（a-z0-9）、`freq`（`1234567890etaoinshrdlucmfwypvbgkjqxz`）、`mixed`（freq + 大写频率序）；也可直接传自定义字符集字符串（≤95 字符）。

### 辅助脚本

```powershell
# 基于 zip 内文件名生成候选（拼音/缩写/中文/年份变体）并离线验证
python scripts\name_crack.py target.zip
```

## 架构

```
src/
├── brute_sha256.cu      SHA-256 破解器（单文件）
│     ├── sha256_compress / sha256_device / sha256_multi   设备端哈希（K 常量内存）
│     ├── brute_kernel_t<L>      编译期定长 L 进位增量枚举（无除法）
│     ├── dict_kernel            字典（定长分桶 + 编译期 L 特化 + Markov 桶序）
│     └── stress_kernel          基准（occupancy 全驻留 + 防 DCE sink）
├── zipcrack.cu          ZIP 破解器
│     ├── ZipCrypto：CRC32 密钥调度 + 12 字节校验头 48 位过滤
│     ├── WinZip AES：PBKDF2-HMAC-SHA1(1000) + 双条目 32 位过滤 + 槽收集
│     └── block-ticket 多长度内核（单启动覆盖 minL..maxL，自动负载均衡）
├── crypto_host.h        主机端密码学库（SHA-1/HMAC/PBKDF2/AES-CTR）
└── zipcrack_cpu.cpp     CPU 参照实现（校验数据硬编码为开发示例，需按目标 zip 自行替换）
experiments/             优化研究原型与报告（REPORT.md 含 12 方向实测）
```

## 关键设计

- **编译期定长特化**：长度 L 作为模板参数，nvcc 常量折叠 padding/W 扩展，SHA-256 路径 +27%、ZipCrypto +8-10%
- **多条目联合过滤**：ZipCrypto 6 条目 = 48 位、AES 双条目 = 32 位，暴力扫描零假阳性
- **AES 假阳性终审**：设备端 16 位 verifier 命中槽收集 → 主机端完整 PBKDF2+HMAC 验证（实测 192/1111 万候选命中，全部正确甄别）
- **Markov 频率序**：`freq` 字符集使常见密码提前命中（实测 xsz2025 命中时间 48.6s → 6.1s）

## 性能与优化历程

详见 `experiments/REPORT.md`：12 个优化方向逐一实测（原子计数器、寄存器溢出、block/thread 扫描、字典布局、位切片可行性、PCIe pinned 上传、hashcat 基准对照等），结论是当前标量实现已接近硬件上限（SHA-256 达 hashcat 的 ~88% per-SM）。

## 安全边界（务必了解）

- **WinZip AES-256 是数学墙**：PBKDF2 每候选需 ~8000 次 SHA-1 压缩，1M 候选/s 已是硬件极限。10 位混合字符密码 ≈ 2.5 万年，暴力不可行——强密码 + AES-256 的安全模型正常工作。
- ZipCrypto 无此防护（每候选仅 ~20 次 CRC 查表），旧格式加密包可被快速破解。

## ⚠️ 伦理声明

本工具仅供**安全研究、教学与合法授权测试**（如找回自己忘记密码的文件）。严禁用于破解他人文件或任何未授权系统。使用者须遵守当地法律法规。

## License

MIT License. 详见 [LICENSE](LICENSE)。
