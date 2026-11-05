# -*- coding: utf-8 -*-
"""放开 ninfer 的 MTP 草稿上限：8 -> 15

背景
----
CLI 写着 `--draft-tokens in [1,8]`，但真正的守门人是 verify batch 的宽度域：

    src/targets/qwen3_6/impl/runtime/text_context_impl.h:685
    if (width > kDFlashDecodeMaximumWidth)   // = 15 + 1 = 16

DFlash 的宽度域被复用成了整个 verify batch 的上限，所以 draft 最多 15
（宽度 = draft + 1 = 16），draft 16 才会真的超。

本脚本改 6 处常量/断言，改完需要重新编译（增量约 20 秒）。

用法
----
    python enable-draft15.py D:\\ninfer\\ninfer-e8-overlay-adaptive
    cd D:\\ninfer && python build_ninfer.py build

注意
----
- 编译前先停掉正在运行的 ninfer-serve.exe，否则链接会报 LNK1104（文件占用）
- draft 15 会掉出 fused 快路径（width 16 不在 [7,10] 内），单流每位置成本从
  0.72ms 涨到 1.08ms；但在 8 路 batch 下摊销更好（0.47ms），净收益 +69%
- 只对极可预测内容有效，日常内容请用 draft 6/8
"""
import os
import re
import shutil
import sys

# (相对路径, 原文本, 新文本, 说明)
EDITS = [
    ("src/ops/wrapper/mtp_round.cpp",
     "if (T < 2 || T > 9) {",
     "if (T < 2 || T > 17) {",
     "host 侧 T 域断言"),

    ("src/ops/wrapper/mtp_round.cpp",
     "mtp_prepare_next_round: T must be in [2,9]",
     "mtp_prepare_next_round: T must be in [2,17]",
     "对应报错文案"),

    ("src/targets/qwen3_6_27b/impl/config.h",
     "kMaximumMtpDraftTokens    = 8;",
     "kMaximumMtpDraftTokens    = 16;",
     "27B 目标的最大草稿数"),

    ("src/product/speculative_options.h",
     "options.draft_tokens > 8",
     "options.draft_tokens > 16",
     "CLI 参数校验上限"),

    ("src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h",
     "kMtpDecodeMaximumDrafts    = 8;",
     "kMtpDecodeMaximumDrafts    = 16;",
     "RoundState 解码帧域"),
]

# gdn_input_proj.cpp 里有 4 处同样的宽度限制，统一改
GDN_FILE = "src/ops/wrapper/gdn_input_proj.cpp"
GDN_OLD = "constexpr std::int32_t kMaximumWidth = 16;"
GDN_NEW = "constexpr std::int32_t kMaximumWidth = 17;"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    root = os.path.abspath(sys.argv[1])
    if not os.path.isdir(root):
        print(f"目录不存在: {root}")
        return 1

    changed = 0
    for rel, old, new, note in EDITS:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            print(f"[跳过] 找不到 {rel}")
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if new in text and old not in text:
            print(f"[已是] {rel} — {note}")
            continue
        if old not in text:
            print(f"[未匹配] {rel} — {note}（源码版本可能不同，请手动确认）")
            continue
        bak = path + ".bak-draft15"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new))
        print(f"[已改] {rel} — {note}")
        changed += 1

    # GDN 宽度限制（可能多处）
    gdn = os.path.join(root, GDN_FILE)
    if os.path.isfile(gdn):
        with open(gdn, encoding="utf-8") as f:
            text = f.read()
        n = text.count(GDN_OLD)
        if n:
            bak = gdn + ".bak-draft15"
            if not os.path.exists(bak):
                shutil.copy2(gdn, bak)
            with open(gdn, "w", encoding="utf-8") as f:
                f.write(text.replace(GDN_OLD, GDN_NEW))
            print(f"[已改] {GDN_FILE} — {n} 处 kMaximumWidth 16 -> 17")
            changed += n
        else:
            print(f"[已是] {GDN_FILE} — kMaximumWidth")

    print(f"\n完成，共 {changed} 处改动。")
    print("下一步：停掉 ninfer-serve.exe，然后 python build_ninfer.py build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
