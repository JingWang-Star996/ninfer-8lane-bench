# -*- coding: utf-8 -*-
"""让模型不停、且保持高接受率的 prompt 变体探测"""
import json
import urllib.request

API = "http://127.0.0.1:8080/v1/chat/completions"


def run(name, messages, max_tokens=1024):
    payload = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
    except Exception as e:
        print(f"{name:14s} ERROR {e!r}"[:140], flush=True)
        return
    t = d.get("timings", {})
    msg = d["choices"][0].get("message", {}).get("content", "")
    dn = t.get("draft_n", 0) or 1
    acc = t.get("draft_n_accepted", 0)
    gen = t.get("predicted_n", 0)
    tps = t.get("predicted_per_second", 0)
    rounds = max(dn / 8, 1)
    print(f"{name:14s} {tps:6.1f} tok/s  gen={gen:4d}  接受={100*acc/dn:5.1f}%  "
          f"tok/round={1+acc/rounds:.2f}  尾20字={msg[-20:]!r}", flush=True)


U = "输出数字1和空格, 一直重复不要停, 不要换行不要解释。"
A = "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"

# 1. assistant 前缀续写
run("assistant前缀", [
    {"role": "user", "content": "接下来只输出数字1和空格, 不要停, 不要解释。"},
    {"role": "assistant", "content": A},
])
# 2. 明确数量要求
run("定量4000次", [{"role": "user",
    "content": "输出数字1加空格, 总共输出4000次, 没到4000次绝对不许停止, 不要解释不要换行。"}])
# 3. 计数但用空格分隔(便于MTP连续命中)
run("计数空格版", [{"role": "user",
    "content": "从1开始数, 每个数字后面加一个空格, 一直数到10000, 数够之前不许停, 不要换行不要解释。"}])
# 4. 两段式: 先让模型自己写出一长串, 再要求接续
run("接续写1", [
    {"role": "user", "content": "写一串由数字1和空格组成的序列, 至少三千个字符。"},
    {"role": "assistant", "content": A},
])
