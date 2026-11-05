# -*- coding: utf-8 -*-
"""8 路纯 decode 聚合吞吐压测 (修正版)

修掉 bench_tps.py 的两个坑:
  1. 固定 8080, 不自动探测其他引擎端口
  2. 吞吐 = 已完成请求的 usage token 总和 / 全体活跃墙钟,
     只统计完整跑完的请求 (半截请求的 usage 不会丢)

用法: python bench_quick.py [seconds=60] [lanes=8] [mode=predict]
  mode=predict: 计数序列, 无思考, 极短上下文 (极限聚合吞吐)
  mode=create : 发散创作, 无思考
"""
import json
import os
import sys
import threading
import time
import urllib.request

API = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-27b"
LOG = "bench_quick_result.txt"   # 结果另存一份到当前目录
MAXTOK = int(os.environ.get("MAXTOK", "2048"))

PROMPT_PREDICT = ("输出数字1加空格, 总共输出4000次, 没到4000次绝对不许停止, 不要解释不要换行。")
PROMPT_CREATE = "写一篇关于深海热泉生态系统的科普长文, 尽量长, 不要停。"

TEMP = {"predict": 0.2, "create": 1.0}


def worker(idx, deadline, out, mode):
    done_tokens = 0
    done_chars = 0
    n_req = 0
    n_done = 0
    first_any = None
    last_any = None
    base = PROMPT_PREDICT if mode == "predict" else PROMPT_CREATE
    while time.time() < deadline - 2.0:
        n_req += 1
        # 每次加唯一编号, 绕开引擎 response checkpoint 复用(否则重复 prompt 会被秒杀)
        prompt = f"{base}\n[L{idx}-R{n_req}-{int(time.time() * 1000) % 99991}]"
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAXTOK,
            "temperature": TEMP[mode],
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            API,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        t_first = None
        t_last = None
        usage_tok = 0
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                for raw in r:
                    body = raw.decode("utf-8", "replace").strip()
                    if not body.startswith("data:"):
                        continue
                    body = body[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        j = json.loads(body)
                    except Exception:
                        continue
                    ch = j.get("choices")
                    if not ch:
                        u = j.get("usage")
                        if u:
                            usage_tok = u.get("completion_tokens", 0)
                        continue
                    d = ch[0].get("delta") or {}
                    txt = d.get("content") or d.get("reasoning_content") or ""
                    if not txt:
                        continue
                    now = time.time()
                    t_last = now
                    if t_first is None:
                        t_first = now
                        if first_any is None:
                            first_any = now
        except Exception as e:
            print(f"  lane{idx} err {e!r}"[:120], flush=True)
            time.sleep(0.3)
            continue
        if t_last and t_first and usage_tok > 0:
            done_tokens += usage_tok
            n_done += 1
            if last_any is None or t_last > last_any:
                last_any = t_last
    span = (last_any - first_any) if (first_any and last_any) else 0.0
    out[idx] = (done_tokens, n_done, span)


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    lanes = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    mode = sys.argv[3] if len(sys.argv) > 3 else "predict"
    deadline = time.time() + seconds
    out = {}
    ths = [threading.Thread(target=worker, args=(i, deadline, out, mode), daemon=True)
           for i in range(1, lanes + 1)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()

    total = sum(v[0] for v in out.values())
    spans = [v[2] for v in out.values() if v[2] > 0]
    reqs = sum(v[1] for v in out.values())
    wall = max(spans) if spans else 0.0
    agg = total / wall if wall > 0.3 else 0.0
    per = sorted([v[0] / v[2] for v in out.values() if v[2] > 0.3])

    lines = []
    lines.append(f"mode={mode} lanes={lanes} seconds={seconds}")
    lines.append(f"completed_requests={reqs} total_tokens={total} wall={wall:.1f}s")
    lines.append(f"AGGREGATE={agg:.1f} tok/s")
    if per:
        mid = per[len(per) // 2]
        lines.append(f"per_lane: min={per[0]:.1f} p50={mid:.1f} max={per[-1]:.1f}")
    body = "\n".join(lines)
    print(body, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(body + "\n---\n")


if __name__ == "__main__":
    main()
