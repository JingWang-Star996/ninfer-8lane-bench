# 1508 tok/s on a single RTX 5090 Laptop — Qwen3.8-27B NVFP4, 8 concurrent lanes

**English** | [简体中文](README.zh-CN.md)

> A single RTX 5090 **Laptop** (24 GB) running a 27B model, 8 concurrent lanes, **1508 tok/s** aggregate.
> Everything here is about the **burst peak** scenario. Daily-throughput numbers are reported separately — do not mix them.

![GPU](https://img.shields.io/badge/GPU-RTX%205090%20Laptop%2024GB-76b900)
![Model](https://img.shields.io/badge/Model-Qwen3.8--27B%20NVFP4-blue)
![Draft](https://img.shields.io/badge/MTP%20draft-15-orange)

---

## TL;DR

| Scenario | Lanes | MTP draft | token/round | Accept | 1 stream | 8-lane end-to-end | 8-lane engine peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Burst peak** | 8 | **15** | 15.98 | 99.9% | 325.4 | **1302.9** | **1508.0** |
| Burst peak | 8 | 8 | 9.00 | 100% | 233.6 | 858.5 | 891.4 |
| Daily, multi-agent | 8 | 6 | 5.90 | 81.8% | — | 465–490 | 784.5 |

Units are tok/s. Same hardware, same model — the only difference between rows is **MTP draft depth**.

The 1508 number in one line:

```
1508 tok/s ≈ 8 lanes × 16 token/round × 11.8 rounds/s
```

Three factors: number of lanes (engine ceiling), tokens committed per round (draft depth × acceptance), round rate (power wall and kernel efficiency).

> ⚠️ **Deeper drafts are not a shortcut to more speed.** Measured: raising draft from 15 to 16 drops the peak
> from 1473 to 924; pushing all the way to 23 gives 1027. A verify width of 16 is the edge of the fast path,
> not a conservative guard. Full data in [results/width-sweep.md](results/width-sweep.md).

---

## Hardware and software

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX 5090 **Laptop**, 24 GB GDDR7, 896 GB/s |
| TGP | 95 W BIOS default, 175 W cap; measured peak **179.86 W** (vendor "beast" profile + OC) |
| SM clock | 2377 MHz peak |
| Temperature | 73 °C (power wall hits before thermal wall) |
| OS / driver | Windows 11, driver 616.56, CUDA 13.4 |
| Engine | [ninfer](https://github.com/Neroued/ninfer) `e8-overlay-adaptive` branch, self-built with MSVC 2022 + CUDA 13.4 |
| Model | Qwen3.8-27B **NVFP4**, 16.78 GiB of weights |
| KV quantization | `rk2v4-e8` |

> ⚠️ This is the **Laptop** part with 24 GB, not the 32 GB desktop card. Do not compare against desktop
> RTX 5090 figures (e.g. 35B-A3B at C=8 reaching 1313.8).

---

## The three measurement definitions

Most doubts come from mixing definitions. Here is where every number comes from.

### 1. Engine peak (`decode=`)

The engine prints one line every 5 seconds; `decode` = decode tokens committed in that window ÷ window length:

```
[2026-11-05 14:22:43.298] throughput interval=5.008s prefill=0.0tok/s decode=1508.0tok/s
                          running=8 prefilling=0 decode_ready=8 waiting=0 avg_decode_batch=8.00
```

That single line satisfies three conditions at once, which rules out the usual objections:

- `prefill=0.0` → no prefill padding, this is **pure decode**
- `avg_decode_batch=8.00` + `waiting=0` → 8 lanes genuinely saturating, nothing queued
- `running=8 / decode_ready=8` → no idle lanes

### 2. End-to-end aggregate

The script sums `usage.completion_tokens` over **all requests that ran to completion**, divided by wall clock
from first token to last token across all active requests.

It comes in about 13% below the engine figure; the gap is TTFT and request-switch idle time.
**This number is harder to game, and it is the one I prefer to quote.**

### 3. Single stream

`timings.predicted_per_second` from the `/v1/chat/completions` response body — straight from the engine.

---

## How the result was reached

### Step 1: the real draft ceiling is 16, not 8

The CLI says `--draft-tokens in [1,8]`, but that is a product-layer hardcode. Follow it down:

```cpp
// src/targets/qwen3_6/impl/runtime/text_context_impl.h:685
if (width > kDFlashDecodeMaximumWidth) {   // = 15 + 1 = 16
    throw std::invalid_argument("target verify batch shape is outside the supported domain");
}
```

**The DFlash width domain is reused as the ceiling for the whole verify batch.** So MTP on this 27B actually
runs `draft 15` (width 16); only draft 16 (width 17) genuinely overflows.

`patches/enable-draft15.py` opens the six constants along that path.

### Step 2: counter-intuitive — slow path + big batch is the sweet spot

The MTP convolution path has a fused fast path whose window is `width <= 3` or `[7,10]`
(`fp8_gdn_conv_plan.cpp:45`). Draft 15 (width 16) necessarily falls onto the Materialized slow path.

Measured single-stream, per-position cost does rise from 0.72 ms to 1.08 ms (+50%). I extrapolated 752 for
8 lanes from that — **wrong**:

| Scenario | Per-token-position cost c |
| --- | ---: |
| Single stream, fused (draft 8) | 0.72 ms |
| Single stream, materialized (draft 15) | 1.08 ms |
| **8-lane batch, materialized (draft 15)** | **0.47 ms** |

The Materialized path's fixed overhead (workspace allocation, weight reads) amortizes across the batch far
better than the fused path does. **Slow path + big batch beats fast path + small batch.**

### Step 3: get acceptance to 99.9%

MTP acceptance depends on how predictable the content is. Measured across prompts:

| Prompt | Accept | token/round | 1 stream |
| --- | ---: | ---: | ---: |
| `输出数字1加空格, 总共输出4000次, 没到4000次绝对不许停止, 不要解释不要换行。` | **99.9%** | **15.98** | **325.4** |
| digits with spaces, repeated (no count constraint) | 84.4% | 7.75 | 203.7 |
| counting up from 1, one per line | 67.9% | 6.43 | 166.7 |
| `abc` loop | 62.5% | 6.00 | 159.6 |
| the letter `a` repeated | 6.2% | 1.50 | 26.2 |

The point: **"100% predictable" is not "100% accepted."** The letter `a` is maximally predictable, but the model
emits one and stops (stop token). A numeric sequence is the most stable. The count constraint ("4000 times")
is what keeps it going.

---

## Reproducing

### 1. Build the engine

This work is based on the `e8-overlay-adaptive` branch of ninfer. **That branch is not merged upstream and has
no public repository**, so this repo ships patches and scripts, not engine source. Bring your own checkout:

```bash
cd /path/to/ninfer            # your ninfer source tree
python build_ninfer.py configure
python build_ninfer.py build
```

### 2. Open the draft ceiling

```bash
python patches/enable-draft15.py /path/to/ninfer
python build_ninfer.py build      # incremental rebuild, ~20 s
```

> Stop any running `ninfer-serve.exe` first, or the link step fails with LNK1104 (file in use).

### 3. Launch the 8-lane profile

```bash
NINFER_DRAFT=15 NINFER_CUDA_GRAPH=1 python build_ninfer.py serve8
```

### 4. Benchmark

```bash
python scripts/bench_quick.py 20 8 predict
```

Watch the engine side at the same time:

```bash
grep "throughput interval" serve.log | grep "running=8"
```

---

## Two traps in your own benchmark script

Easy to hit when you write your own; the results will be off by a large factor.

### Trap 1: response-checkpoint reuse kills the request

Send the same prompt repeatedly and the engine recognises an identical prefix; with
`reuse=restore_response_checkpoint` it hits the stop token immediately and emits 2 tokens:

```
[req 614] done finish=stop_token gen=2 cache=40 reuse=restore_response_checkpoint
```

**Fix**: append a unique suffix per request (already done in the script).

### Trap 2: the model stops on its own

Counting stops at 292; short CJK sentences stop after one. Assistant prefill does not help either (`gen=2`).

**Fix**: put an explicit count requirement in the prompt.

Hit both traps and your numbers land at one fifth of reality.

---

## Can it go faster? Two failed attempts

After 1508 I tried two routes. Both hit walls. Documented so nobody repeats the work.

### Attempt 1: 12 concurrent lanes (failed)

Eight lanes is the engine's hard ceiling. Twelve requires changing the GDN replay row geometry, and four walls
refuse in sequence: `--max-concurrency must be in [1,8]` → `GDN replay record capacity exceeds eight rows`
(`kMaximumRecordCapacity = 8`) → `kMaximumRows = 8` → the kernel only registers an 8-row fold geometry.
More importantly, **the arithmetic does not pay**: 12 lanes works out to roughly 983, still under 1000, so
digging into the kernel was not worth it.

### Attempt 2: draft depth to 23 (failed, with counter-intuitive data)

Fitting draft 8 and draft 15 gave `round time = 75.5 ms + 0.59 ms × positions`, which extrapolates draft 23 to
2143 tok/s. So I opened four layers of limits (CLI cap, GQA workspace, RoundState window, GDN width). It
compiled, draft 23 launched, acceptance 98.7%, 23.5 tok/round — **and then it ran at half speed**:

| draft | verify width | Engine peak | End-to-end | token/round | Round time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 16 | **1473.5** | 1199.1 | 15.98 | 87 ms |
| 16 | 17 | 897.1 | 760.5 | 16.78 | 145 ms |
| 19 | 20 | 988.8 | 866.8 | — | 147 ms |
| 23 | 24 | 1026.7 | 757.0 | 23.50 | 183 ms |

The cause is `kSmallTChunkTokens = 6`: width 16 is exactly three chunks (6+6+4), an optimised tiling. Past 16 the
route changes, per-round fixed cost jumps from 85 ms to 145 ms, and the slow route's marginal cost is 5.4 ms per
position — nine times the fast path's 0.59 ms. **Deeper drafts do commit more tokens (1.84×), but each round
costs 2.1× more. Net loss.**

There is a hidden cost too: width 24 needs a larger workspace, and the previous `--kv-capacity 94208` fails to
allocate (97 MB short), so it had to drop to 86016. **Deeper drafts are paid for in context length — at a loss.**

### What the two failures tell us

1. **Around 1500 is the ceiling of this kernel family.** Neither deeper drafts nor more lanes move it.
2. **The only remaining lever is the ~75 ms fixed cost per round** (weights 20 ms + verify 9 ms + scheduling /
   workspace ~55 ms), not tokens per round. Under the same model, halving that fixed cost would put width 16 near
   2700 — **but that is another extrapolation, never measured. Do not quote it.** The direction is credible (fixed
   cost dominates); the number is not.
3. **Stop extrapolating from local linear fits.** 0.59 ms is the slope of one fast path; cross the boundary and it
   is void. This is the same class of error as the corrections below.

---

## Numbers I got wrong before

Part of the doubt was self-inflicted. Corrections below.

| What I said | What it actually was | Real value |
| --- | --- | ---: |
| "8 lanes, 1200 / 1269 tok/s" | total tokens ÷ a **nominal 12 s**, while requests actually ran 29–103 s | **489.6** |
| "8 lanes, 780" | a 5-second engine window peak, presented as sustained throughput | ~**485** sustained |
| "deeper drafts reach 2143" | two-point linear extrapolation, both anchors inside the fast path | **1026.7** |

The first was a plain calculation error: the script divided by the nominal duration, inflating 3–10×. The second
was instantaneous-vs-average confusion — 780 does exist, but as a 5-second sample, not a 30-second mean.

**What I do now**: every public number carries its definition (`engine peak` / `end-to-end` / `single stream`),
and peaks are reported separately from sustained rates.

---

## Limitations (important)

1. **This is a burst scenario only.** 8 lanes × 16 tokens/round requires near-perfectly predictable content (99.9%
   acceptance). Real chat, code and agent work do not get there: ~60–70 tok/s per lane, **465–490 tok/s** for 8.
2. **Draft 15 is worse than draft 8 on ordinary content.** Once acceptance drops, verifying a deep draft is pure
   waste. Use 6 or 8 day to day.
3. **Eight lanes is the engine hard limit.** Twelve needs GDN replay row geometry changes (`kMaximumRows = 8`) and
   CUDA kernel work — not done here.
4. **Depends on an unmerged branch.** Any upstream update invalidates these conclusions.
5. **The laptop bottleneck is the power wall.** At the 95 W default, 8 lanes gives ~580 tok/s — 2.6× below the
   179 W result. That is a TGP limit, not cooling.
6. **1500 is this kernel's ceiling, not a tuning problem.** Past verify width 16 you fall off the fast path;
   deeper drafts and more lanes are both negative. Getting further means changing scheduling and CUDA Graph
   coverage, not flags.

---

## Data and evidence

- `results/raw-serve-log.txt` — key log lines with timestamps
- `results/bench-output.txt` — benchmark script output
- `results/configs.md` — full launch parameters per profile
- `results/width-sweep.md` — the draft 15→23 experiment in full
- `results/width-sweep-raw/` — raw engine logs on both sides of the cliff (draft 16 / draft 23)
- `scripts/bench_quick.py` — benchmark script
- `scripts/probe_prompts.py` — acceptance probe
- `scripts/sweep_draft.py` — restart-per-depth sweep, archives raw logs
- `patches/enable-draft15.py` — the six constants
- `scripts/patch_w24.py` — the draft 23 experiment patch (reversible; the experiment failed, kept for the record)

---

## A note on the environment

The engine is a self-built `ninfer-serve` (native Windows, not vLLM or llama.cpp). The reason is that every
parameter is exposed on the command line and VRAM can be planned to the megabyte — decisive under a 24 GB
ceiling. The cost is dependence on a branch that is not upstream.

---

## License

MIT (covers this repository's scripts and documents only).
