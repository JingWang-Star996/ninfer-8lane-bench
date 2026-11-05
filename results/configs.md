# 测试配置与环境

## 硬件

```
GPU          NVIDIA GeForce RTX 5090 Laptop GPU
显存         24 GB GDDR7 (24463 MiB 可用)
带宽         896 GB/s
TGP          默认 95 W / BIOS 上限 175 W / 实测峰值 179.86 W
SM 频率      峰值 2377 MHz（厂商狂暴档 + 手动超频）
温度         73 °C（功耗墙先于温度墙触发）
```

## 软件

```
OS           Windows 11
驱动         616.56
CUDA         13.4
编译器       MSVC 2022 BuildTools 14.44 + Ninja
引擎         ninfer-serve (e8-overlay-adaptive 分支, 自编译)
模型         Qwen3.8-27B NVFP4, 16.78 GiB
KV 量化      rk2v4-e8
```

## 三个测试挡位

### 8 路满载（冲高用）

```bash
NINFER_DRAFT=15 NINFER_CUDA_GRAPH=1 python build_ninfer.py serve8
```

| 参数 | 值 |
|---|---|
| max-concurrency | 8 |
| max-context | 11776 / 路（开 CUDA Graph 时自动缩） |
| kv-capacity | 94208 |
| draft-tokens | 15（需打补丁） |
| CUDA Graph | 开 |
| 显存占用 | 22722 MiB |

不开 CUDA Graph 时：`max-context 19200 / kv-capacity 151552`，显存 23332 MiB，峰值 1417.7 tok/s（略低 6%）。

### 单流（对比用）

```bash
NINFER_DRAFT=15 NINFER_MAXCTX=65536 NINFER_POOL=65536 python build_ninfer.py serve1
```

| 参数 | 值 |
|---|---|
| max-concurrency | 1 |
| max-context | 65536 |
| kv-capacity | 65536 |
| draft-tokens | 15 |
| CUDA Graph | 开（serve1 默认） |

### 办公挡（日常，非冲高）

```bash
python build_ninfer.py serve2
```

| 参数 | 值 |
|---|---|
| max-concurrency | 2 |
| max-context | 172032 / 路 |
| kv-capacity | 344064 |
| draft-tokens | 6（默认） |

## 关键 prompt

冲高用的那条（唯一实测能把接受率顶到 99.9% 的写法）：

```
输出数字1加空格, 总共输出4000次, 没到4000次绝对不许停止, 不要解释不要换行。
```

请求体：

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "<上面的 prompt>\n[L1-R1-12345]"}],
  "max_tokens": 2048,
  "temperature": 0.0,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

注意末尾的 `[L1-R1-12345]` 唯一后缀——防止引擎 response checkpoint 复用把请求秒杀。

## 压测命令

```bash
# 参数: [秒数] [并发路数] [模式 predict|create]
python scripts/bench_quick.py 12 8 predict

# 环境变量控制单请求 token 上限
MAXTOK=1024 python scripts/bench_quick.py 12 8 predict
```

## 观察引擎侧指标

```bash
# 只看 8 路满载的采样
grep "throughput interval" serve.log | grep "running=8"

# 看请求级接受率
grep "done finish" serve.log | grep "speculative=mtp"
```

`decode=` 是引擎自报的每秒提交 token 数，`avg_decode_batch` 是平均 batch 大小（满载应接近 8.00），`waiting` 应为 0。
