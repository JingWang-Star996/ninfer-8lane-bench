# -*- coding: utf-8 -*-
"""draft 深度扫描：逐个深度重启引擎 → 跑同一压测 → 记录峰值与 tok/round。

    python scripts/sweep_draft.py 15 16 17 19 21 23

依赖：ninfer-serve.exe、模型 artifact、scripts/bench_quick.py 都在默认路径。
用环境变量覆盖：

    NINFER_EXE    默认 D:\\ninfer\\build_manual\\apps\\ninfer-serve.exe
    NINFER_MODEL  默认 D:\\ninfer\\models\\qwen3_8_27b_nvfp4.ninfer
    NINFER_PORT   默认 8080

每个深度的引擎日志会落到 results/width-sweep-raw/serve-draftNN.log，
这样「峰值」和「tok/round」都能事后核对，而不是只有一行汇总。
"""
import os
import re
import subprocess
import sys
import time

EXE = os.environ.get("NINFER_EXE", r"D:\ninfer\build_manual\apps\ninfer-serve.exe")
MODEL = os.environ.get("NINFER_MODEL", r"D:\ninfer\models\qwen3_8_27b_nvfp4.ninfer")
PORT = os.environ.get("NINFER_PORT", "8080")
BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_quick.py")
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "width-sweep-raw")
LOG = os.path.join(os.environ.get("TEMP", "."), "sweep_serve.log")

BASE = [
    "--kv-dtype", "rk2v4-e8",
    "--prefill-chunk", "1024",
    "--spec", "mtp",
    "--lm-head-draft",
    "--pending-timeout-ms", "300000",
    "--max-concurrency", "8",
    "--max-context", "11776",
    "--kv-capacity", "86016",   # 宽度 24 的 workspace 更大，94208 会 OOM
    "--host", "127.0.0.1",
    "--port", PORT,
]


def stop_engine():
    subprocess.run(["taskkill", "/F", "/IM", "ninfer-serve.exe"], capture_output=True)
    time.sleep(3)


def start_engine(draft):
    log = open(LOG, "w", encoding="utf-8", errors="replace")
    cmd = [EXE, MODEL, "--draft-tokens", str(draft)] + BASE
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                         cwd=os.path.dirname(EXE))
    for _ in range(120):
        time.sleep(1)
        if p.poll() is not None:
            return None, open(LOG, encoding="utf-8", errors="replace").read()
        if "listening on" in open(LOG, encoding="utf-8", errors="replace").read():
            return p, ""
    return p, "timeout"


def bench():
    r = subprocess.run([sys.executable, BENCH, "16", "8", "predict"],
                       capture_output=True, text=True, timeout=240,
                       cwd=os.path.dirname(BENCH))
    return r.stdout


results = []
for draft in [int(x) for x in sys.argv[1:]]:
    stop_engine()
    proc, txt = start_engine(draft)
    if proc is None:
        err = [l for l in txt.splitlines() if "error" in l.lower()][:1]
        print(f"draft={draft:2d}  启动失败: {err}")
        results.append((draft, None, None, None))
        continue

    time.sleep(3)
    out = bench()
    time.sleep(2)

    log = open(LOG, encoding="utf-8", errors="replace").read()
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, f"serve-draft{draft:02d}.log"), "w",
              encoding="utf-8") as f:
        f.write(log)

    peaks = [float(m) for m in re.findall(r"decode=([\d.]+)tok/s", log)]
    tpr = re.findall(r"([\d.]+)tok/round", log)
    agg = re.search(r"AGGREGATE=([\d.]+)", out)
    peak = max(peaks) if peaks else 0.0
    print(f"draft={draft:2d}  引擎峰值={peak:7.1f}  "
          f"端到端={agg.group(1) if agg else '?':>7}  "
          f"tok/round={tpr[0] if tpr else '?'}")
    results.append((draft, peak, agg.group(1) if agg else None, tpr[0] if tpr else None))
    proc.kill()
    time.sleep(2)

print("\n深度  峰值     端到端    tok/round")
for d, pk, ag, tp in results:
    print(f"{d:3d}  {pk if pk else '失败':>8}  {ag or '':>7}  {tp or ''}")
