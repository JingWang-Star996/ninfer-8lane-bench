# 冲高实验：把 draft 深度从 15 推到 23（失败）

> 结论先说：**verify 宽度 16 是快速路径的边界，不是保守校验。** 越过它，每轮耗时从 85 ms 跳到 145 ms
> 再爬到 183 ms，多出来的 token 全部被吃掉，8 路峰值从 1473 掉到 900–1030。
> 本仓库此前基于两点线性外推得到的「draft 23 = 2143 tok/s」由此被实测证伪。

实验日期 2026-11-05，硬件/模型与主 README 相同（RTX 5090 Laptop 24 GB，Qwen3.8-27B NVFP4，
`rk2v4-e8` KV，8 路，`--max-context 11776`）。为让各深度可对比，本实验统一把 `--kv-capacity`
降到 86016（原因见 §4），因此 draft 15 在这里测到 1473.5 / 1383.9，而不是主 README 的 1508.0
（后者用 94208）。

---

## 1. 动机

主 README 的两个实测点（draft 8 → 891.4，draft 15 → 1508.0）拟合出

```
每轮耗时 T = 75.5 ms + 0.59 ms × 位置数
```

拟合显示每多提交一个位置只多花 0.59 ms，而多拿一个 token。按此外推，位置 24（draft 23）
应达 2143 tok/s。这次实验就是去验证这个外推。

---

## 2. 四道墙

想把 draft 从 15 推上去，先要放开四层限制。前三层是编译期常量，第四层是**几何**：

| # | 位置 | 现象 |
|---|---|---|
| 1 | `src/product/speculative_options.h` | `--spec mtp requires --draft-tokens in [1,16]` |
| 2 | `src/ops/wrapper/gqa_attention.cpp` `kMaximumVerifyTokens = 16` | 启动即报 `gqa_attention workspace: invalid profile or interval` |
| 3 | `round_state.h` `kMtpDecodeMaximumDrafts = 16` / `kDFlashDecodeMaximumDrafts = 15` | 启动即报 `RoundState MTP draft window exceeds the decode frame domain` |
| 4 | `gdn_input_proj.cpp` `kMaximumWidth = 17`（4 处）、`replay.cpp` `width > 16`（2 处） | 通过前面的墙后才会碰到 |

把 1–4 全部放开（共 11 处替换，见 `scripts/patch_w24.py`），增量编译约 60 秒，draft 23 可以正常启动。
所以这四层**不是**真正的约束，只是门。

---

## 3. 真正的约束：快速路径的分块几何

`gqa_attention.cpp` 里 `kSmallTChunkTokens = 6`，batch > 1 时走 `ChunkedSmallT`：

| 宽度 | 分块 | 是否在快速路径 |
|---|---|---|
| 16 | 6 + 6 + 4（3 块） | ✅ |
| 17 | 6 + 6 + 5（3 块，但越界） | ❌ 换路 |
| 24 | 6 × 4（4 块） | ❌ 换路 |

分块数一旦变化，同一条 round 的固定开销就重算——这正是下面数据里 85 ms → 145 ms 的跳变来源。

---

## 4. 实测数据

同一压测脚本（`scripts/bench_quick.py 16 8 predict`）、同一配置，逐深度重启引擎测量。

| draft | verify 宽度 | 引擎峰值 | 端到端 | tok/round | 接受率 | 轮/秒 | 每轮耗时 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **15** | 16 | **1473.5 / 1383.9** | 1199.1 / 1202.7 | 15.98 | 99.9% | 11.5 | 87 ms |
| 16 | 17 | 897.1 / 923.8 | 760.5 / 809.5 | 16.78 | 99.3% | 6.9 | 145 ms |
| 17 | 18 | 949.7 | 738.9 | — | — | 7.1 | 141 ms |
| 19 | 20 | 988.8 | 866.8 | — | — | 6.8 | 147 ms |
| 21 | 22 | 982.6 | 727.9 | — | — | 6.7 | 149 ms |
| 23 | 24 | 1026.7 | 757.0 | 23.50 | 98.7% | 5.5 | 183 ms |

（draft 15 / 16 各测两次，给两次的值；其余各测一次。峰值取引擎 5 秒窗口 `decode=` 的最大值。）

### 关键读法

**接受率没有崩。** draft 16 的 `16.78 tok/round (99.3%)`、draft 23 的 `23.50 tok/round (98.7%)`
都接近理论满额——草稿质量完全没问题。崩的是**每轮耗时**：

```
宽度 16 → 85 ms      每位置边际 0.59 ms
宽度 17 → 145 ms     跳变 +60 ms（换路，不是线性增长）
宽度 24 → 183 ms     慢路上的边际 ≈ 5.4 ms/位置，是快速路径的 9 倍
```

**更深 = 更贵。** 位置 24 每轮能提交 23.5 个 token，是位置 16 的 1.84 倍，但每轮耗时是 2.1 倍，
净亏 12%。

### 附带的显存代价

宽度 24 需要更大的 verify workspace。原来 `--kv-capacity 94208` 在启动时直接失败：

```
[error] ninfer-serve: requested Engine runtime reservation requires 6362365184 bytes,
        but only 6260344064 bytes are available for runtime capacity
```

只差 97 MB，但必须降到 86016 才能起来。**更深的草稿要拿上下文去换**，这是额外的隐性成本。

---

## 5. 被证伪的外推

| 来源 | 位置 20 | 位置 24 |
|---|---|---|
| 两点线性外推（实验前） | 1834 | 2143 |
| 实测 | 988.8（draft 19） | 1026.7（draft 23） |

误差 2 倍以上。原因：外推的两个锚点（位置 9、16）都落在**同一条**快速路径上，斜率 0.59 ms 是那条
路径的斜率。穿过路径边界后斜率变了 9 倍，任何基于旧斜率的推算都会系统性高估。

这条写在这里，是因为它和主 README「我此前报错过的数字」是同一类错误：**拿局部线性关系去推一个已经
换路的系统**。

---

## 6. 对下一步的意义

1. **1500 附近是这套 kernel 家族的天花板。** 加草稿深度这条路已经封死，不要再用它做外推。
2. **剩下唯一有效的杠杆是每轮那约 75 ms 的固定开销**，不是 token/round。
   主 README 里 85 ms 的构成是：权重读取约 20 ms + 验证约 9 ms + 其余约 55 ms 的启动/同步/workspace。
   按同一模型估算，固定开销砍半、位置 16 能到约 2700——**但这同样是外推，没有实测**。
   方向（固定开销是大头）有实测锚点支撑，具体数字没有。真要往这儿走，得改调度与 CUDA Graph 覆盖。
3. **12 路并发的路同样走不通**（`kMaximumRows = 8` 的 GDN replay 几何），但它和深度是两条独立的墙：
   宽度卡验证，行数卡并发。

---

## 7. 复现

```bash
# 1. 打实验补丁（自动留 .bak-w24 备份）
python scripts/patch_w24.py /path/to/ninfer

# 2. 编译（必须先停掉 ninfer-serve.exe，否则 LNK1104）
python build_ninfer.py build

# 3. 逐深度重启 + 压测，原始日志存到 results/width-sweep-raw/
python scripts/sweep_draft.py 15 16 17 19 21 23

# 4. 还原（恢复 15/16 常量并重新编译）
python scripts/patch_w24.py /path/to/ninfer revert
```

原始日志：`results/width-sweep-raw/serve-draft16.log`（悬崖内侧）、`serve-draft23.log`（悬崖外侧）。
两份都保留了 `throughput interval` 与 `speculative=mtp ... tok/round` 原文，可直接核对。
