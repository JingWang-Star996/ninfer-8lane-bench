# -*- coding: utf-8 -*-
"""把 verify 宽度上限从 16 推到 24（draft 23）的实验补丁。

    python scripts/patch_w24.py /path/to/ninfer            # 应用
    python scripts/patch_w24.py /path/to/ninfer revert     # 还原

设计成「可逆」的：每处改动第一次应用前都会备份为 <file>.bak-w24，
revert 时从备份原样恢复并删除备份。跑完实验请务必 revert 并重新编译，
否则引擎会停在慢路径上（峰值约 900–1030，而不是 1470+）。

实验结论见 results/width-sweep.md：这条路是负收益，宽度 16 才是快速路径边界。
"""
import os
import shutil
import sys

EDITS = [
    # 1. CLI 层：--draft-tokens 上限
    ("src/product/speculative_options.h",
     [("options.draft_tokens > 16", "options.draft_tokens > 23")]),
    # 2. GQA workspace：batch>1 时宽度上限
    ("src/ops/wrapper/gqa_attention.cpp",
     [("constexpr std::int32_t kMaximumVerifyTokens          = 16;",
       "constexpr std::int32_t kMaximumVerifyTokens          = 24;")]),
    # 3. RoundState：MTP / DFlash 两个 draft 窗口
    ("src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h",
     [("inline constexpr std::uint32_t kMtpDecodeMaximumDrafts    = 16;",
       "inline constexpr std::uint32_t kMtpDecodeMaximumDrafts    = 23;"),
      ("inline constexpr std::uint32_t kDFlashDecodeMaximumDrafts = 15;",
       "inline constexpr std::uint32_t kDFlashDecodeMaximumDrafts = 23;")]),
    # 4. GDN 输入投影宽度（4 处同名常量）
    ("src/ops/wrapper/gdn_input_proj.cpp",
     [("constexpr std::int32_t kMaximumWidth = 17;",
       "constexpr std::int32_t kMaximumWidth = 24;")]),
    # 5. GDN replay fold 的宽度校验（2 处）
    ("src/ops/linear_attention/gated_delta_net/replay.cpp",
     [("width < 2 || width > 16", "width < 2 || width > 24"),
      ("spec.width < 2 || spec.width > 16", "spec.width < 2 || spec.width > 24")]),
    # 6. 27B target 的 MTP 常量
    ("src/targets/qwen3_6_27b/impl/config.h",
     [("inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 16;",
       "inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 24;")]),
    # 7. mtp_round 的 T 窗口
    ("src/ops/wrapper/mtp_round.cpp",
     [("if (T < 2 || T > 17) {", "if (T < 2 || T > 24) {"),
      ('"mtp_prepare_next_round: T must be in [2,17]"',
       '"mtp_prepare_next_round: T must be in [2,24]"')]),
]


def apply(src):
    for rel, subs in EDITS:
        path = os.path.join(src, rel)
        text = open(path, encoding="utf-8").read()
        bak = path + ".bak-w24"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        hit = 0
        for old, new in subs:
            n = text.count(old)
            if n == 0:
                print(f"  !! 未命中 {rel}: {old[:52]}")
                continue
            text = text.replace(old, new)
            hit += n
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print(f"  ok {rel}  ({hit} 处)")


def revert(src):
    for rel, _ in EDITS:
        path = os.path.join(src, rel)
        bak = path + ".bak-w24"
        if os.path.exists(bak):
            shutil.copy2(bak, path)
            os.remove(bak)
            print(f"  restored {rel}")
        else:
            print(f"  -- 无备份 {rel}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "revert":
        revert(root)
    else:
        apply(root)
