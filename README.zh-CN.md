# 1508 tok/s on a single RTX 5090 Laptop — Qwen3.8-27B NVFP4, 8 concurrent lanes

[English](README.md) | **简体中文**

> 单张 RTX 5090 **Laptop**（24 GB）跑 27B 模型，8 路并发聚合 **1508 tok/s**。  
> 本文只讨论**极限脉冲场景**，日常吞吐数据同时给出，请勿混淆口径。

![GPU](https://img.shields.io/badge/GPU-RTX%205090%20Laptop%2024GB-76b900)

![Model](https://img.shields.io/badge/Model-Qwen3.8--27B%20NVFP4-blue)

![Draft](https://img.shields.io/badge/MTP%20draft-15-orange)



---

## TL;DR

| 场景        | 并发 | MTP draft | token/轮 | 接受率   | 单流    | 8 路端到端     | 8 路引擎峰值    |
| --------- | -- | --------- | ------- | ----- | ----- | ---------- | ---------- |
| **极限脉冲**  | 8  | **15**    | 15.98   | 99.9% | 325.4 | **1302.9** | **1508.0** |
| 极限脉冲      | 8  | 8         | 9.00    | 100%  | 233.6 | 858.5      | 891.4      |
| 日常多 Agent | 8  | 6         | 5.90    | 81.8% | —     | 465~490    | 784.5      |

单位 tok/s。三行的差别全部来自 **MTP 草稿深度**，硬件和模型完全相同。

一句话解释 1508：

```
1508 tok/s ≈ 8 路 × 16 token/轮 × 11.8 轮/s
```

三个因子分别对应：并发路数（引擎上限）、每轮提交 token 数（草稿深度 + 接受率）、轮速（受功耗墙和 kernel 效率约束）。

> ⚠️ **别把「加深草稿」当成继续加速的捷径。** 实测把 draft 从 15 推到 16，峰值从 1473 掉到 924；
> 继续推到 23 也只有 1027。verify 宽度 16 是快速路径的边界，不是保守校验。
> 完整数据见 [results/width-sweep.md](results/width-sweep.md)。

---

## 硬件与软件

| 项       | 值                                                                                |
| ------- | -------------------------------------------------------------------------------- |
| GPU     | NVIDIA RTX 5090 **Laptop** GPU，24 GB GDDR7，896 GB/s                              |
| TGP     | BIOS 默认 95 W，上限 175 W；实测峰值 **179.86 W**（厂商狂暴档 + 超频）                              |
| SM 频率   | 峰值 2377 MHz                                                                      |
| 温度      | 73 °C（功耗墙先于温度墙）                                                                  |
| OS / 驱动 | Windows 11，驱动 616.56，CUDA 13.4                                                   |
| 引擎      | [ninfer](https://github.com/Neroued/ninfer) `e8-overlay-adaptive` 分支，MSVC 2022 + CUDA 13.4 自编译 |
| 模型      | Qwen3.8-27B **NVFP4**，权重 16.78 GiB                                               |
| KV 量化   | `rk2v4-e8`                                                                       |

> ⚠️ 注意这是 **Laptop** 24 GB，不是桌面版 32 GB。厂商文档里 RTX 5090 桌面版的数据（例如 35B-A3B 在 C=8 跑到 1313.8）不能直接对比。

---

## 三个口径的定义

质疑通常来自口径混用。这里把每一个数字的来源写死。

### 1. 引擎峰值 `decode=`

引擎每 5 秒打印一行，`decode` 字段 = 该窗口内已提交的 decode token 总数 ÷ 窗口秒数：

```
[2026-11-05 14:22:43.298] throughput interval=5.008s prefill=0.0tok/s decode=1508.0tok/s
                          running=8 prefilling=0 decode_ready=8 waiting=0 avg_decode_batch=8.00
```

这一行同时满足三个条件，可以直接排除常见质疑：

- `prefill=0.0` → 不含 prefill 水分，是**纯 decode**
- `avg_decode_batch=8.00` + `waiting=0` → 8 路**真实满载**，没有请求排队
- `running=8 / decode_ready=8` → 没有空转

### 2. 端到端聚合

脚本统计**所有完整跑完**的请求的 `usage.completion_tokens` 总和 ÷ 全体活跃墙钟时间（首 token 到末 token）。

比引擎值低 13% 左右，差额是 TTFT 和请求切换间隙。**这个数字更难刷，也是我更愿意对外引用的那个。**

### 3. 单流

`/v1/chat/completions` 返回体里的 `timings.predicted_per_second`，引擎直出。

---

## 这个成绩是怎么来的

### 关键一：找到真正的 draft 上限是 16，不是 8

引擎 CLI 写着 `--draft-tokens in [1,8]`，但那是产品层的硬编码。往下追：

```cpp
// src/targets/qwen3_6/impl/runtime/text_context_impl.h:685
if (width > kDFlashDecodeMaximumWidth) {   // = 15 + 1 = 16
    throw std::invalid_argument("target verify batch shape is outside the supported domain");
}
```

**DFlash 的宽度域被复用成了整个 verify batch 的上限。** 所以 27B 的 MTP 实际能跑 `draft 15`（宽度 16），draft 16（宽度 17）才会真的超。

`patches/enable-draft15.py` 就是放开这条链路上的 6 处常量。

### 关键二：反直觉——慢路径 + 大 batch 才是甜点

MTP 的卷积路径有个 fused 快路径，窗口只有 `width <= 3` 或 `[7,10]`（源码 `fp8_gdn_conv_plan.cpp:45`）。draft 15（width 16）必然掉到 Materialized 慢路径。

按单流测，每位置成本确实从 0.72 ms 涨到 1.08 ms（慢 50%）。我据此推算 8 路只有 752，**错了**：

| 场景                                   | 每 token 位置成本 c |
| ------------------------------------ | -------------- |
| 单流 fused（draft 8）                    | 0.72 ms        |
| 单流 materialized（draft 15）            | 1.08 ms        |
| **8 路 batch materialized（draft 15）** | **0.47 ms**    |

Materialized 路径的固定开销（workspace 分配、权重读取）在 batch 维度摊薄得比 fused 更彻底。**慢路径 + 大 batch 反而比快路径 + 小 batch 更快。**

### 关键三：让接受率达到 99.9%

MTP 的接受率取决于内容可预测性。实测各种 prompt 的对比：

| prompt                                          | 接受率       | token/轮   | 单流        |
| ----------------------------------------------- | --------- | --------- | --------- |
| `输出数字1加空格, 总共输出4000次, 没到4000次绝对不许停止, 不要解释不要换行。` | **99.9%** | **15.98** | **325.4** |
| 数字加空格重复（不加数量约束）                                 | 84.4%     | 7.75      | 203.7     |
| 从 1 开始递增计数（每行一个）                                | 67.9%     | 6.43      | 166.7     |
| `abc` 循环                                        | 62.5%     | 6.00      | 159.6     |
| 字母 `a` 重复                                       | 6.2%      | 1.50      | 26.2      |

要点：**"100% 可预测"不等于"100% 接受率"**。字母 `a` 是最可预测的，但模型输出一个就自己停了（stop token），反而是数字序列最稳。数量约束（"4000 次"）是让它不停的关键。

---

## 复现

### 1. 编译引擎

本项目基于 ninfer 的 `e8-overlay-adaptive` 分支。**该分支尚未合入上游、也未公开仓库**，因此本仓库不提供引擎源码，只提供补丁与完整的复现脚本。你需要自备该分支源码：

```bash
cd /path/to/ninfer            # 你的 ninfer 源码目录
python build_ninfer.py configure
python build_ninfer.py build
```

### 2. 放开 draft 上限

```bash
python patches/enable-draft15.py /path/to/ninfer
python build_ninfer.py build      # 重新编译，约 20 秒（增量）
```

> 编译前必须停掉正在运行的 `ninfer-serve.exe`，否则链接会因文件占用失败（LNK1104）。

### 3. 启动 8 路挡

```bash
NINFER_DRAFT=15 NINFER_CUDA_GRAPH=1 python build_ninfer.py serve8
```

### 4. 压测

```bash
python scripts/bench_quick.py 20 8 predict
```

同时观察引擎侧：

```bash
grep "throughput interval" serve.log | grep "running=8"
```

---

## 关于压测脚本的两个坑

自己写脚本复现时，很容易踩到，结果会偏得离谱。

### 坑 1：response checkpoint 复用会把请求秒杀

同一个 prompt 连发，引擎判定为相同前缀，`reuse=restore_response_checkpoint` 后立刻命中 stop token，一次只输出 2 个 token：

```
[req 614] done finish=stop_token gen=2 cache=40 reuse=restore_response_checkpoint
```

**对策**：每次请求带唯一后缀（脚本里已做）。

### 坑 2：模型会主动停

数字递增数到 292 个就停，CJK 短句输出一次就停。用 assistant prefill 也没用（`gen=2`）。

**对策**：prompt 里写死数量要求。

踩了这两个坑，测出来的数字会低到 1/5。

---

## 还能不能更快：两次失败的探索

1508 之后我试过两条路，都撞墙了。写在这里，避免别人再花同样的时间。

### 探索一：加并发到 12 路（失败）

八路是引擎的硬上限，12 路需要改 GDN replay 的行数几何，四道墙依次拒绝：
`--max-concurrency must be in [1,8]` → `GDN replay record capacity exceeds eight rows`
（`kMaximumRecordCapacity = 8`）→ `kMaximumRows = 8` → kernel 只注册了 8 行折叠几何。
更关键的是**算下来不划算**：12 路的理论值约 983，仍不到 1000，所以没有再往 kernel 里钻。

### 探索二：加深草稿到 23（失败，且有反直觉数据）

拿 draft 8 与 15 两个点拟合出「每轮耗时 = 75.5 ms + 0.59 ms × 位置数」，按此外推 draft 23 应有
2143 tok/s。于是放开四层限制（CLI 上限、GQA workspace、RoundState 窗口、GDN 宽度），
编译通过、draft 23 正常启动、接受率 98.7%、23.5 tok/round——**然后慢了一半**：

| draft | verify 宽度 | 引擎峰值 | 端到端 | tok/round | 每轮耗时 |
|---:|---:|---:|---:|---:|---:|
| 15 | 16 | **1473.5** | 1199.1 | 15.98 | 87 ms |
| 16 | 17 | 897.1 | 760.5 | 16.78 | 145 ms |
| 19 | 20 | 988.8 | 866.8 | — | 147 ms |
| 23 | 24 | 1026.7 | 757.0 | 23.50 | 183 ms |

原因在 `kSmallTChunkTokens = 6`：宽度 16 正好是 3 块（6+6+4），是优化过的分块；越过 16 就换路，
每轮固定开销从 85 ms 跳到 145 ms，慢路径上每位置边际成本 5.4 ms，是快速路径（0.59 ms）的 9 倍。
**更深的草稿确实能提交更多 token（1.84 倍），但每轮要多花 2.1 倍时间，净亏。**

还多了一个隐性成本：宽度 24 的 workspace 更大，原 `--kv-capacity 94208` 直接 OOM（差 97 MB），
得降到 86016。**更深的草稿是拿上下文换的，而且是负收益。**

### 这两次失败给出的结论

1. **1500 附近是这套 kernel 家族的天花板。** 加深草稿、加并发都动不了它。
2. **剩下唯一有效的杠杆是每轮那 75 ms 的固定开销**（权重 20 ms + 验证 9 ms + 调度/workspace 约 55 ms），
   而不是 token/round。按同一模型估算，固定开销砍半、位置 16 能到约 2700——**但这同样是一次外推，没测过，
   别当结论引用**。方向可信（固定开销是大头），数字不可信。
3. **不要再拿局部线性关系外推。** 0.59 ms 是同一条快速路径的斜率，跨过路径边界就失效了。
   这和下面「我此前报错过的数字」是同一类错误。

---

## 我此前报错过的数字

质疑有一部分是我自己造成的，这里一并更正。

| 曾对外说                    | 实际含义                                   | 当时的真实值      |
| ----------------------- | -------------------------------------- | ----------- |
| "8 路 1200 / 1269 tok/s" | 总 token ÷ **标称 12 秒**，而请求实际跑了 29~103 秒 | **489.6**   |
| "8 路 780"               | 引擎 5 秒窗口峰值，被当成持续吞吐                     | 持续约 **485** |
| "加深草稿能到 2143"            | 两点线性外推，锚点都在快速路径内，跨过边界后斜率变 9 倍          | **1026.7**  |

第一个是纯计算错误：脚本把标称时长当分母，虚高 3~10 倍。第二个是瞬时值和平均值的混淆——780 确实存在，但它是一个 5 秒窗口的采样，不是 30 秒的平均。

**现在的做法**：所有对外数字同时标注口径（`引擎峰值` / `端到端` / `单流`），峰值和持续分开报。

---

## 局限（重要）

1. **这只是极限场景。** 8 路 × 16 token/轮需要内容几乎完全可预测（99.9% 接受率）。真实对话、代码、agent 任务达不到，日常每路 60~70 tok/s，8 路聚合 **465~490 tok/s**。
2. **draft 15 在普通内容上不如 draft 8。** 接受率掉了之后，深草稿的验证开销纯属浪费。日常请用 draft 6/8。
3. **8 路是引擎硬上限。** 想上 12 路需要改 GDN replay 的行数几何（`kMaximumRows = 8`），涉及 CUDA kernel，本项目未做。
4. **依赖未合入上游的分支。** 上游更新后所有结论都要重新验证。
5. **笔记本的瓶颈是功耗墙。** 默认 95 W 时 8 路只有约 580 tok/s，和 179 W 差 2.6 倍。这不是散热问题，是 TGP 限制。
6. **1500 是这套 kernel 的天花板，不是配置问题。** verify 宽度 16 之后掉出快速路径，加深草稿、加并发都是负收益（见上文两次探索）。想再往上，得改调度和 CUDA Graph 覆盖，不是调参能解决的。

---

## 数据与证据

- `results/raw-serve-log.txt` — 关键日志原文（含时间戳）
- `results/bench-output.txt` — 压测脚本输出
- `results/configs.md` — 各挡位完整启动参数
- `results/width-sweep.md` — 冲高实验（draft 15→23）完整记录与结论
- `results/width-sweep-raw/` — 悬崖两侧的引擎原始日志（draft 16 / draft 23）
- `scripts/bench_quick.py` — 压测脚本
- `scripts/probe_prompts.py` — 接受率探测
- `scripts/sweep_draft.py` — 逐深度重启 + 压测，自动归档原始日志
- `patches/enable-draft15.py` — 6 处改动
- `scripts/patch_w24.py` — draft 23 实验补丁（可逆；实验失败，仅作记录）

---

## 环境说明

引擎是自编译的 `ninfer-serve`（Windows 原生，非 vLLM/llama.cpp）。选它的原因是参数全部暴露在命令行、显存可精确到 MB 规划——对 24 GB 这种紧约束是决定性的。代价是依赖尚未合入上游的分支。

---

## License

MIT（仅覆盖本仓库的脚本与文档）
