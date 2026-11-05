# -*- coding: utf-8 -*-
"""寻找能让 MTP 接受率最高、且不被 stop 截断的 prompt"""
import json
import urllib.request

API = "http://127.0.0.1:8080/v1/chat/completions"

CASES = [
    ("数字空格重复", "只输出数字1和空格, 一直重复不要停, 不要换行, 不要解释, 不要输出别的: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"),
    ("字母a重复", "只输出小写字母a, 不要空格不要换行不要标点, 一直重复不要停, 不要解释: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
    ("abc循环", "只输出 abc 三个字母的循环, 不要空格不要换行, 一直重复不要停: abcabcabcabcabcabcabcabcabcabcabcabc"),
    ("计数", "从1开始每行输出一个递增数字, 一直数下去, 不要停。"),
    ("重复短句", "输出下面这句话, 每行一遍, 一直重复不要停, 不要编号: 好的。"),
]


def run(name, prompt):
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    t = d.get("timings", {})
    msg = d["choices"][0].get("message", {}).get("content", "")
    dn = t.get("draft_n", 0) or 1
    acc = t.get("draft_n_accepted", 0)
    gen = t.get("predicted_n", 0)
    tps = t.get("predicted_per_second", 0)
    print(f"{name:12s} {tps:6.1f} tok/s  gen={gen:4d}  接受率={100*acc/dn:5.1f}%  "
          f"tok/round={1+acc/max(dn/8,1):.2f}  首40字={msg[:40]!r}", flush=True)


for n, p in CASES:
    try:
        run(n, p)
    except Exception as e:
        print(f"{n} ERROR {e!r}"[:150], flush=True)
