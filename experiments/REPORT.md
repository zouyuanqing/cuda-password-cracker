# CUDA SHA-256 密码爆破 — 算法级效率优化研究报告

**硬件**: RTX 5070 Laptop GPU（Blackwell sm_120，36 SM，8GB） · **CUDA 13.2** · **Windows**
**对象**: `brute_sha256.cu`（~600 行）
**日期**: 2026-08 · **所有原型位于 `experiments\` 子目录，主文件未被修改**

---

## 0. 执行摘要（TL;DR）

| 结论 | 一句话 |
|---|---|
| **基线 78 GH/s 是假象** | `stress_kernel` 的哈希结果 `h[]` 从不被读取，`-O3` 把它整段死代码消除（ptxas 报 10 寄存器、0 栈帧）。78 GH/s 测的是**全局 atomicAdd 吞吐**，不是哈希吞吐。 |
| **真实单块 SHA-256 吞吐 ≈ 5.1 GH/s** | 编译期定长 L 的进位枚举路径实测 5.07–5.14 GH/s；`brute_kernel7` 全扫描实测 **5.065 GH/s**。 |
| **已接近 hashcat 水平** | hashcat mode 1400 在 RTX 5070(48SM) = 7.71 GH/s ≈ 160.6 MH/s/SM；本项目 5.06 GH/s/36SM ≈ 140.7 MH/s/SM ≈ **88% hashcat per-SM**。 |
| **核心是计算受限（ALU 饱和）** | 25%→75% occupancy 吞吐恒定 ~5.1 GH/s；估计 ~2200–2400 整数指令/hash，已达 INT32 峰值约 88%。 |
| **可落地的优化只有少数几项** | ① 全部路径改编译期 L（~27%）；② 进位枚举泛化（~8%）；③ pinned 上传（2.1×）；④ 字典长度分桶（~1.8×）；⑤ Markov 排序（命中时间量级改善）。 |
| **不做的** | 位切片 SHA-256（ARX 无 S-box，hashcat 也不做）、block/thread 调优、`#pragma unroll`、`-maxrregcount`、早停比较（已最优）、atomicAdd 微调（~2%）。 |

---

## 1. 环境与基线复现

```powershell
# 编译（全程无管道捕获 nvcc 输出；C4819 为无害编码警告）
nvcc -O3 -arch=sm_120 -Xptxas -v brute_sha256.cu -o build\brute_sha256.exe
# 运行
.\build\brute_sha256.exe --stress 3 7                    # stress 基线
.\build\brute_sha256.exe --dict candidates.txt <target>   # 字典基线
.\build\brute_sha256.exe 7 abcdefghijklmnopqrstuvwxyz ABCDEFG --progress  # 暴力基线
```

实测基线（复现成功）：

| 模式 | 结果 | 说明 |
|---|---|---|
| stress（原） | **78.17 GH/s** | 见方向 1 —— 哈希被 DCE，测的是 atomicAdd |
| 真实单块（编译期 L=7） | **5.065 GH/s**（`brute_kernel7` 全扫描 8.03e9 次 / 1.586s） | 真实哈希吞吐 |
| dict（SoA） | **2.447 GH/s**（1434 万行 / 5.86ms） | 真实字典吞吐 |
| CPU 单线程 | ~5.2 MH/s | 约 970× 差距 |

`nvcc -Xptxas -v` 关键信息（全部 kernel **0 spill**）：

| kernel | 寄存器 | 栈帧 | 结论 |
|---|---|---|---|
| `stress_kernel` | **10** | 0 | 哈希被 DCE（铁证） |
| `brute_kernel7` | 64 | 0 | 定长 L，w[] 完全寄存器驻留 |
| `brute_kernel`（runtime-L） | 48 | 96B | chunk[64]+buf[32] 落 local memory |
| `dict_kernel` | 60 | 64B | 同上 |

---

## 2. 12 个优化方向逐一评估

> 标注约定：**[实测]** = 有原型与数据；**[理论]** = 文献/架构分析；**[已最优]** = 实测证明无需改动。

### 方向 1 — 每哈希全局 atomicAdd 计数器 **[实测]**

**原理**：`stress/brute/dict_kernel` 每哈希对单一 `ops_counter` 地址 `atomicAdd(1)`，用户怀疑 16.7M 线程争用单地址是瓶颈。

**实测**（`exp1_atomic.cu`，3 秒/变体）：

| 变体 | 吞吐 | 说明 |
|---|---|---|
| A 原始（h 未用 + 每哈希 atomicAdd） | 78.27 GH/s | **哈希被 DCE**，纯 atomicAdd 吞吐 |
| B 哈希 sink，**无** 每迭代原子 | 4.006 GH/s | 真实哈希吞吐（runtime-L） |
| C 哈希 sink + 全局 atomicAdd/哈希 | 3.933 GH/s | 原子开销仅 **1.8%** |
| D 哈希 sink + 共享内存 atomicAdd/哈希 | 3.916 GH/s | 共享原子无额外收益 |

**结论**：atomicAdd 的吞吐上限（~78 GH/s）远超真实哈希吞吐（~5 GH/s），**不是瓶颈**，去掉它只省 ~2%。真正的问题是 **stress 基准本身测错了东西**（见下方 diff 建议 ①）。

**是否实施**：计数改 per-thread/per-block 聚合（一次 atomic）——可选、低优先级、~2%。**难度**：低。

---

### 方向 2 — brute_kernel 的 div/mod vs 进位增量枚举 **[实测]**

**原理**：`brute_kernel`（通用）每候选做 L 次 64-bit 除法+取模（`x % N; x /= N`），`brute_kernel7` 已用进位增量（add+compare+sub，无除法）。方向：把进位增量泛化到任意 L。

**实测**（`exp2_enum.cu`，N=26，编译期 L，单位 GH/s）：

| 方法 | L=5 | L=6 | L=7 | L=8 |
|---|---|---|---|---|
| carry（进位增量） | 5.111 | 5.112 | 5.074 | 5.048 |
| div/mod（除法取模） | 4.820 | 4.668 | 4.674 | 4.581 |
| **carry 优势** | **+6.0%** | **+9.5%** | **+8.6%** | **+10.2%** |

**结论**：进位增量带来稳定 **~8–10%** 收益，且与 L 无关。`brute_kernel7` 已做，`brute_kernel`（通用路径）未做——泛化是免费收益。

**是否实施**：**是**（模板化 unrolled digit counter，见 diff ②）。**难度**：低–中。

---

### 方向 3 — w[64] 数组寄存器溢出 **[实测，已最优]**

**原理**：怀疑 `w[64]`（256B）溢出到 local memory。

**实测**：`-Xptxas -v` 显示**所有 kernel 0 spill stores/loads**。定长 L 路径 w[] 完全寄存器驻留（40 regs / 0 栈帧）；runtime-L 路径的 96B 栈是 `chunk[64]+buf[32]`（这是"按字节构消息"的 local memory，非寄存器溢出）。`#pragma unroll` 变体（`sha256_compress_u`）与基础版**编译结果一致**（exp2 中 carry+device == carry+fixed，40 regs / 0 spill）。

**结论**：**无寄存器溢出**。`#pragma unroll`、显式 w0..w63、`__launch_bounds__`、`-maxrregcount` 均无必要（编译器已充分优化）。

**是否实施**：否。**难度**：N/A。

---

### 方向 4 — blocks/threads 配置调优 **[实测，已最优]**

**实测**（`exp4_sweep.cu`，编译期 L=7，1 秒/配置）：

| threads | 最优 blocks/SM | GH/s |
|---|---|---|
| 64 | 6 | 5.070 |
| 128 | 4 | 5.068 |
| 256 | 4/6 | 5.089 |
| **512** | **1** | **5.122** |

全平台 **5.06–5.12 GH/s**，几乎与配置无关；**512 线程/1 block/SM（25% occupancy）就饱和 ALU**。这是**计算受限（ALU 吞吐受限）而非延迟受限**的铁证——再多 warp 也提不上去。

**结论**：当前 256 线程 / 6 blocks/SM 已近最优，无需调。

**是否实施**：否。**难度**：N/A。

---

### 方向 5 — 字典模式访存布局 **[实测]**

**原理**：SoA（flat+offsets+lens）间接寻址疑似破坏合并；候选定长 padded / uint4 向量化 / 按长度分桶。

**实测**（`exp5_dict.cu`，1434 万行全扫描）：

| 布局 | 时间 | 吞吐 |
|---|---|---|
| SoA（flat+offsets+lens），256 线程 | 6.57 ms | 2.18 GH/s |
| **Padded 64B 定长 + uint4 合并加载** | **7.30 ms** | **1.97 GH/s（更差）** |

**关键洞察**：
1. **SoA 并不坏**——`offsets` 单调递增，相邻线程读相邻 `flat` 区间，flat 访问天然近似顺序，合并性尚可。
2. **padding 有害**——64B 定长把 125MB 膨胀到 915MB（7×），合并加载省下的时间抵不过 7× 访存增量。
3. 字典 2.45 GH/s ≈ 计算峰值 5.1 GH/s 的一半，瓶颈是 **runtime-L 哈希路径 + 变长分支**，不是布局。

**真正有效的做法**：**按长度分桶**（主机端按 `len` 分桶，每个桶用编译期 L 内核 `sha256_fixed<L>` 扫描）。同时复用了方向 2/7 的"编译期 L"特化，预期 2.45 → ~4.5 GH/s（~1.8×），且不膨胀内存。

**是否实施**：长度分桶**是**；padding **否**。**难度**：中。

---

### 方向 6 — 位切片（bitsliced）SHA-256 **[理论，不建议]**

**结论：死路。** SHA-256 是纯 ARX（Add-Rotate-Xor，**无 S-box**）结构。位切片的价值在于用 N 位列逻辑并行化 S-box/布尔密码（如 DES）；对 ARX 而言位切片**反而增加**指令数与寄存器压力（8 个状态词 × 32 bit-plane = 仅状态就 256 寄存器），且加法进位链无法消除。因此：
- **hashcat 只对 DES 族位切片**（`KERN_ATTR_BITSLICE` 仅出现在 m1500/m3000/m14000）；SHA-256（mode 1400）用**标量 per-thread 全展开**实现。
- MD4/MD5/SHA-1 在 hashcat 中是**向量化**（u32x，SIMD-within-thread），**也不是位切片**。
- 未检索到任何"位切片 SHA-256"学术论文；bitcoin 挖矿优化（Courtois et al. 2014）用的是 midstate + 消息调度复用，非位切片。

**预期收益**：零或负。**是否实施**：否。**难度**：高。

---

### 方向 7 — 固定长度单块 W 预展开/常量折叠 **[实测]**

**原理**：7 字符小写下，w[0..15] 结构固定（pad 0x80、长度常量 0x38），w[2..14]=0 可常量折叠，只有 w[0]、w[1] 随候选变化。

**实测**（`exp2_enum.cu`）：`sha256_fixed<L>`（模板 L，显式 `#pragma unroll` 的 padding/w 构建）与 `sha256_device(buf, L, h)`（L 为模板常量）**编译到完全相同的 40 regs / 0 栈帧**，吞吐一致（5.06 vs 5.07 GH/s）。**nvcc 已经自动做了常量折叠**。

**真正的收益点**不在"手写折叠"，而在 **L 是否编译期已知**：

| 路径 | 吞吐 | 栈帧 |
|---|---|---|
| runtime-L（`brute_kernel`/`dict_kernel`/`stress_kernel` 的通用 `sha256_device`） | **4.0 GH/s** | 96B |
| compile-time-L（`brute_kernel7` / 模板化） | **5.1 GH/s** | 0 |

即 **"编译期 L 特化"= ~27% 收益**，编译器已代为完成折叠。动作是把所有路径模板化（见 diff ②③）。

**是否实施**：与方向 2/5 合并实施。**难度**：中。

---

### 方向 8 — 早停比较 **[理论，已最优]**

**分析**：当前逐字比较、命中即 `break`。首个字 `h[0]==target[0]` 命中概率 = 2⁻³²，故 99.9999999% 的候选只比较 **1 个字**。已等价于"先比 1 字、命中才全比"的最优策略，无浪费。

**是否实施**：否（已最优）。**难度**：N/A。

---

### 方向 9 — PCIe/上传（pinned + async + 双流）**[实测]**

**实测**（`exp6_pcie.cu`）：

| 传输 | pageable | pinned (cudaHostAlloc) | 提升 |
|---|---|---|---|
| 240MB（flat 125 + offsets 57 + lens 57） | 13.36 GB/s = **18.8 ms** | 28.5 GB/s = **8.8 ms** | **2.1×** |
| 125MB（flat 单块） | 9.78 ms | 4.62 ms | 2.1× |

**关键量化**：字典单次上传 ~19ms（pageable）**是内核全扫描 ~6ms 的 3 倍**。所以：
- **单目标字典攻击**：上传占主导，pinned 内存直接把端到端时间从 ~25ms 降到 ~15ms，收益显著。
- **多目标/常驻复用**：上传摊销，收益可忽略。
- `cudaMemcpyAsync` + 双流对"一次性上传"收益有限（无并发计算可重叠）；真正的收益来自 pinned 本身。

**是否实施**：**是**（简单，字典加载改 `cudaHostAlloc`）。**难度**：低。

---

### 方向 10 — 候选排序（Markov / best64 规则）**[理论，独立章节]**

**原理**：不改吞吐，改**命中顺序**。hashcat 的 `best64.rule` 与 Markov 统计按"真实口令中字符转移概率"重排字典/暴力空间，使常见口令排前。

**评估**：
- 对字典：按 Markov 概率降序重排 1434 万行，平均命中时间可降 **1–2 个数量级**（取决于目标分布，命中目标越靠前收益越大）。
- 对暴力：Markov 链可重排"每位的字符概率 + 位数枚举顺序"，等效于对 26^L 空间做概率排序。
- 实现成本在**主机端/预处理**，不涉及 GPU kernel 改动；可离线训练（如用 rockyou 自身统计 1–3 阶 Markov 转移）。
- **不要求 GPU 原型**；作为"平均破解时间"维度与吞吐维度并列的价值主张。

**是否实施**：是（独立功能模块）。**难度**：中。

---

### 方向 11 — 参考基准（hashcat mode 1400）**[实测 + 文献]**

| GPU | SM 数 | SHA-256 (mode 1400) | per-SM |
|---|---|---|---|
| RTX 4090 (128 SM) | 128 | 21.79 GH/s | 170.2 MH/s/SM |
| RTX 5090 (170 SM) | 170 | 27.68 GH/s | 162.8 MH/s/SM |
| RTX 5070 (48 SM) | 48 | 7.71 GH/s | 160.6 MH/s/SM |
| **本项目 5070 Laptop** | **36** | **5.065 GH/s（实测）** | **140.7 MH/s/SM** |

来源：hashcat forum thread-13327、tid=11277 RTX 4090、5090 基准。
⚠️ 注意：tutorials.technology 的"4090 SHA-256 = 63 GH/s"是 **SHA-1 误标**（真实 4090 SHA-1 = 50.9 GH/s）。

**结论**：本项目标量实现已达 hashcat 的 **~88% per-SM**。剩余 ~12% 差距来自 hashcat 的向量化/ILP 交织/手写微调，属"最后 10%"的工程投入，收益有限。

---

### 方向 12 — Blackwell sm_120 特性 **[理论]**

- **统一 FP32/INT32 核**（128/SM），INT32 吞吐相对 Ada **翻倍**——SHA-256 是 INT-pipe 受限，直接受益（本项目的 5.1 GH/s 已吃到这波红利）。来源：Blackwell whitepaper、Blackwell Integer 讨论。
- **寄存器文件 256KB/SM、48 warps/SM、255 regs/thread、128KB SMEM**（Blackwell Tuning Guide v13.3）——对寄存器驻留的 SHA-256 均非约束。
- **张量核（tcgen05/TMEM/UMMA）在 sm_120（GeForce）不可用，且对 SHA-256 无适用性**（确认，无 mma 可用）。
- **rotr 微优化**：本项目裸 idiom `(x>>n)|(x<<(32-n))` 已被 nvcc 编译为单条 `SHF`；`cuda::std::rotl/rotr` 在 CUDA ≤13.3 会有多余 `and.b32 r,31`，本项目未用，无此问题。可选改用 `__funnelshift_l` / `asm("shf.l.wrap.b32")`，收益 <1%，属"最后 1%"。

**是否实施**：无新指令可用；不实施。**难度**：N/A。

---

## 3. 实施路线图（按 收益/难度 排序）

| 优先级 | 优化 | 预期收益 | 难度 | 类型 |
|---|---|---|---|---|
| **P0** | 修 `stress_kernel` 的 DCE bug（h 结果 sink） | 基准变正确（不提升吞吐） | 低 | 正确性 |
| **P1** | 进位枚举模板化到任意 L（方向 2 + 7） | **~8%（暴力）** | 低–中 | 算法 |
| **P2** | pinned 内存上传（方向 9） | **2.1×（单目标字典端到端）** | 低 | 传输 |
| **P3** | 字典按长度分桶 + 编译期 L 内核（方向 5） | **~1.8×（字典）** | 中 | 算法 |
| **P4** | Markov / best64 候选排序（方向 10） | 命中时间 1–2 个量级 | 中 | 算法（独立） |
| — | atomicAdd 聚合、block 调优、unroll、早停 | ~2% 或 0 | 低 | 不必要 |
| — | 位切片 SHA-256 | 0/负 | 高 | **不做** |

**核心结论**：这是一个**已经接近硬件上限的标量实现**。算法层面唯一有实质意义的提升是"编译期 L 特化"（已在 L=7 落地，需泛化）与"字典长度分桶"；超出 ~5 GH/s 需要重写为向量化/手工 SASS（hashcat 级工程），预期上限 ~10%（到 ~5.6 GH/s），位切片则不可行。

---

## 4. 原型文件清单（`experiments\`）

| 文件 | 内容 | 编译命令 |
|---|---|---|
| `sha256_common.cuh` | 共享 SHA-256 核心（base / `#pragma unroll` / 模板定长 `sha256_fixed<L>`） | （头文件） |
| `exp1_atomic.cu` | atomicAdd 成本 + DCE 验证（变体 A/B/C/D） | `nvcc -O3 -arch=sm_120 experiments\exp1_atomic.cu -o experiments\exp1_atomic.exe` |
| `exp2_enum.cu` | carry vs div/mod × L=5..8；device vs fixed 哈希路径 | `nvcc -O3 -arch=sm_120 -Xptxas -v experiments\exp2_enum.cu -o experiments\exp2_enum.exe` |
| `exp4_sweep.cu` | block/thread 扫描（64/128/256/512 × blocks/SM） | `nvcc -O3 -arch=sm_120 experiments\exp4_sweep.cu -o experiments\exp4_sweep.exe` |
| `exp5_dict.cu` | 字典 SoA vs padded 64B 布局 | `nvcc -O3 -arch=sm_120 experiments\exp5_dict.cu -o experiments\exp5_dict.exe` |
| `exp6_pcie.cu` | pageable vs pinned H2D 上传 | `nvcc -O3 -arch=sm_120 experiments\exp6_pcie.cu -o experiments\exp6_pcie.exe` |

运行参数：
```powershell
.\experiments\exp1_atomic.exe 3 7 abcdefghijklmnopqrstuvwxyz 256
.\experiments\exp2_enum.exe 2 26
.\experiments\exp4_sweep.exe 1
.\experiments\exp5_dict.exe candidates.txt
.\experiments\exp6_pcie.exe
```

> 所有原型用 `clock64()` 自计时（`goal = clock_khz * 1000 * duration`），单变体 ~1–3 秒 GPU 时间，总计 <1 分钟。

---

## 5. 对 `brute_sha256.cu` 的精确 diff 建议（仅描述，未实际修改）

### ①【P0 正确性】修复 `stress_kernel` 的死代码消除

**问题**：`stress_kernel` 中 `h[8]` 写入后从不读取，`-O3` 消除整段 SHA-256，78 GH/s 是 atomicAdd 吞吐。

**改法**：把哈希结果接入一个不可被证明为死值的 sink：
```cuda
// 循环外声明 uint32_t acc = 0;
uint32_t h[8];
sha256_device(buf, L, h);
acc += h[0];                      // 新增：强制计算哈希
atomicAdd(ops_counter, 1ULL);
// 循环结束后：if (acc == 0xDEADBEEFu) atomicAdd(ops_counter, 0x1ULL);  // 防 DCE
```
同时可把每哈希 `atomicAdd` 改为 per-thread 计数、循环末一次 `atomicAdd`（方向 1，~2%）。

### ②【P1】`brute_kernel` 泛化为模板化进位枚举

把 `brute_kernel`（div/mod 版本）替换为编译期 L 的进位增量模板，与 `brute_kernel7` 同构：
```cuda
template<int L>
__global__ void brute_kernel_t(int N, uint64_t start, uint64_t limit,
                               const uint32_t* target, long long* found_idx,
                               char* found_pwd, unsigned long long* ops_counter) {
    uint64_t tid = start + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t end = start + limit;
    unsigned int d[L]; uint64_t t = tid;
#pragma unroll
    for (int i = 0; i < L; ++i) { d[i] = (unsigned int)(t % (uint64_t)N); t /= (uint64_t)N; }
    char buf[L];
    while (tid < end) {
        if (*found_idx != -1) return;
#pragma unroll
        for (int i = 0; i < L; ++i) buf[i] = d_alphabet[d[i]];
        uint32_t h[8]; sha256_fixed<L>(buf, h);     // 编译期 L，取代 sha256_device(buf, L, h)
        // ...比较 + atomicCAS 写入 + 进位增量（同 brute_kernel7）...
        unsigned int carry = 1;
#pragma unroll
        for (int i = 0; i < L; ++i) { unsigned int s = d[i] + carry; if (s >= (unsigned int)N) { s -= (unsigned int)N; carry = 1; } else carry = 0; d[i] = s; }
        tid += 1;   // 每线程连续段
    }
}
```
在 `main` 里按 `L` 分发（`switch(L){case 5: brute_kernel_t<5><<<...>>>...; ...}`）。预期 4.7 → 5.1 GH/s（~8%，且与 L 无关）。注意：进位枚举要求每线程处理**连续**索引段，与现有 strided 不同，需相应调整起始/步长映射。

### ③【P3】字典按长度分桶 + 编译期 L 内核

`dict_kernel` 改为：主机端按 `len` 分桶（桶内保存 flat 内偏移，保持紧凑、不 padding），对每个长度桶调用 `dict_kernel_fixed<L>`（用 `sha256_fixed<L>`）。桶 1..55 用单块快速路径，>55 用 `sha256_multi`（保留）。预期 2.45 → ~4.5 GH/s。

### ④【P2】字典上传改 pinned

`main` 的 `--dict` 分支字典缓冲区改用 `cudaHostAlloc(..., cudaHostAllocDefault)` 加载 + `cudaMemcpyAsync` 到默认流（或直接 `cudaMemcpy`，pinned 已含 2.1× 收益）。单目标攻击端到端 ~25ms → ~15ms。

### ⑤【可选】进度计数聚合

三个 kernel 的每哈希 `atomicAdd(ops_counter,1)` 改为 per-thread/per-block 累加、周期一次性 `atomicAdd`（~2%，非关键路径）。

---

## 6. 已验证实测 vs 理论分析标注

| 方向 | 状态 |
|---|---|
| 1 atomicAdd 成本 / DCE 假象 | ✅ **实测**（exp1） |
| 2 div/mod vs 进位枚举 | ✅ **实测**（exp2） |
| 3 w[64] 无溢出 | ✅ **实测**（`-Xptxas -v`，全 kernel 0 spill） |
| 4 block/thread 平台 | ✅ **实测**（exp4） |
| 5 字典布局（padding 更差） | ✅ **实测**（exp5） |
| 5' 长度分桶 ~1.8× | ⚠️ **理论**（由 4.0→5.1 GH/s 特化收益外推，未单独建桶实测） |
| 6 位切片不可行 | ✅ **文献**（hashcat 实现 + ARX 分析） |
| 7 编译期 L 折叠 | ✅ **实测**（exp2，device==fixed） |
| 8 早停已最优 | ✅ **理论**（概率论证，2⁻³²） |
| 9 pinned 2.1× | ✅ **实测**（exp6） |
| 10 Markov 排序 | ⚠️ **理论**（无 GPU 原型，按任务要求） |
| 11 hashcat 基准 | ✅ **文献**（hashcat 论坛/第三方基准） |
| 12 Blackwell 特性 | ✅ **文献**（官方 whitepaper/tuning guide） |

---

## 7. 附注：主文件发现的两个次要 bug（非性能，建议一并修复）

1. **非 progress 暴力路径计时恒为 0**：`main` 中 `cudaEventDestroy(start/stop)` 在 `cudaEventElapsedTime(&ms, start, stop)` **之前**执行，导致 `--progress` 之外路径打印 `time(s)=0 hashes/s=0`。（`--progress` 路径计时正常，故本报告暴力基线用 `--progress` 测得。）
2. **stress 计时漂移**：`goal = clock_khz * 1000 * duration` 用基频 1545MHz 计算，但 GPU 实际 boost 到 ~2850MHz，`--stress 3` 实际只跑 ~1.6s。不影响吞吐比值，但"secs"打印有误导。

---

## 8. 结论

1. **当前实现不是"没优化好"，而是"已接近标量 SHA-256 的硬件上限"**：5.06 GH/s ≈ 88% hashcat per-SM，ALU 饱和（25% occupancy 即封顶）。
2. **唯一有意义的算法级提升**是"编译期 L 特化"的普及化（泛化进位枚举 + 字典长度分桶），合计把暴力 ~8%、字典 ~1.8×。
3. **"78 GH/s" 与 "atomicAdd 是瓶颈" 两个原始假设均被证伪**——前者是 DCE 假象，后者实测仅 ~2%。
4. 超出 ~5.6 GH/s 需要 hashcat 级向量化/手写 SASS 工程投入；位切片对 ARX 型 SHA-256 不可行。
5. 对"平均破解时间"最有价值的独立优化是 **Markov/best64 候选排序**（方向 10），与吞吐优化正交。
