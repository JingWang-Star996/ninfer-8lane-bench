# -*- coding: utf-8 -*-
"""把报告里的表格和代码块渲染成 PNG，供 Word 插入"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT, exist_ok=True)

YAHEI = "C:/Windows/Fonts/msyh.ttc"
YAHEI_B = "C:/Windows/Fonts/msyhbd.ttc"
MONO = "C:/Windows/Fonts/consola.ttf"

DPI_SCALE = 2          # 2 倍渲染，插入 Word 后清晰
INK = (28, 28, 30)
MUTED = (96, 100, 106)
LINE = (208, 212, 218)
HEAD_BG = (240, 242, 245)
ZEBRA = (250, 251, 252)
CODE_BG = (246, 248, 250)
ACCENT = (176, 58, 46)


def f(path, size):
    return ImageFont.truetype(path, int(size * DPI_SCALE))


def draw_table(name, header, rows, widths, font_size=13, title=None, note=None,
               highlight_col=None):
    """widths 为相对列宽"""
    head_font = f(YAHEI_B, font_size)
    body_font = f(YAHEI, font_size)
    pad = int(11 * DPI_SCALE)
    total_w = int(sum(widths) * DPI_SCALE)
    col_w = [int(w * DPI_SCALE) for w in widths]

    # 先量行高
    line_h = int(font_size * DPI_SCALE * 1.55)
    row_h = line_h + pad * 2
    title_h = int(34 * DPI_SCALE) if title else 0
    note_h = int(26 * DPI_SCALE) if note else 0
    height = title_h + row_h * (len(rows) + 1) + note_h + pad

    img = Image.new("RGB", (total_w + pad * 2, height), "white")
    d = ImageDraw.Draw(img)

    y = pad // 2
    if title:
        d.text((pad, y + int(4 * DPI_SCALE)), title, font=f(YAHEI_B, font_size + 3), fill=INK)
        y += title_h

    # 表头
    d.rectangle([pad, y, pad + total_w, y + row_h], fill=HEAD_BG)
    x = pad
    for i, cell in enumerate(header):
        d.text((x + pad // 2, y + pad), cell, font=head_font, fill=INK)
        x += col_w[i]
    y += row_h

    # 数据行
    for ri, row in enumerate(rows):
        if ri % 2 == 1:
            d.rectangle([pad, y, pad + total_w, y + row_h], fill=ZEBRA)
        x = pad
        for ci, cell in enumerate(row):
            color = ACCENT if (highlight_col is not None and ci == highlight_col) else INK
            d.text((x + pad // 2, y + pad), cell, font=body_font, fill=color)
            x += col_w[ci]
        y += row_h

    # 网格线
    gy = pad // 2 + title_h
    for i in range(len(rows) + 2):
        yy = gy + i * row_h
        d.line([pad, yy, pad + total_w, yy], fill=LINE, width=max(1, DPI_SCALE // 2))
    gx = pad
    d.line([pad, gy, pad, y], fill=LINE, width=max(1, DPI_SCALE // 2))
    for w in col_w:
        gx += w
        d.line([gx, gy, gx, y], fill=LINE, width=max(1, DPI_SCALE // 2))

    if note:
        d.text((pad, y + int(6 * DPI_SCALE)), note, font=f(YAHEI, font_size - 2), fill=MUTED)

    path = os.path.join(OUT, name)
    img.save(path, "PNG")
    print("table ->", path, img.size)


def has_cjk(s):
    return any("\u2e80" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef" for ch in s)


def draw_code(name, lines, font_size=12.5, title=None, highlight_lines=None):
    mono = f(MONO, font_size)
    cjk = f(YAHEI, font_size * 1.02)   # Consolas 没有中文字形，含中文的行改用雅黑
    pad = int(14 * DPI_SCALE)
    line_h = int(font_size * DPI_SCALE * 1.62)
    widths = [cjk.getlength(l) if has_cjk(l) else mono.getlength(l) for l in lines]
    w = int(max(widths or [400])) + pad * 2
    title_h = int(30 * DPI_SCALE) if title else 0
    h = line_h * len(lines) + pad * 2 + title_h

    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, h], fill=CODE_BG)

    y = pad
    if title:
        d.text((pad, y - int(4 * DPI_SCALE)), title, font=f(YAHEI_B, font_size - 0.5), fill=MUTED)
        y += title_h

    hl = set(highlight_lines or [])
    for i, line in enumerate(lines):
        color = ACCENT if i in hl else (32, 34, 38)
        d.text((pad, y), line, font=(cjk if has_cjk(line) else mono), fill=color)
        y += line_h

    path = os.path.join(OUT, name)
    img.save(path, "PNG")
    print("code  ->", path, img.size)


# ---------- 表格 ----------

draw_table(
    "t1-result.png",
    ["场景", "并发", "draft", "token/轮", "接受率", "端到端", "引擎峰值"],
    [
        ["极限脉冲", "8", "15", "15.98", "99.9%", "1302.9", "1508.0"],
        ["极限脉冲", "8", "8", "9.00", "100%", "858.5", "891.4"],
        ["日常多 Agent", "8", "6", "5.90", "81.8%", "465~490", "784.5"],
    ],
    [150, 62, 62, 88, 78, 96, 96],
    title="表 1  三种配置对比（单位 tok/s，硬件与模型完全相同）",
    note="峰值取自引擎 5 秒窗口采样；端到端为所有完成请求的 token 总和除以活跃墙钟。",
    highlight_col=6,
)

draw_table(
    "t2-prompt.png",
    ["prompt 写法", "接受率", "token/轮", "单流 tok/s"],
    [
        ["数字 1 加空格，限定 4000 次", "99.9%", "15.98", "325.4"],
        ["数字加空格重复（无数量约束）", "84.4%", "7.75", "203.7"],
        ["从 1 递增计数（每行一个）", "67.9%", "6.43", "166.7"],
        ["abc 循环", "62.5%", "6.00", "159.6"],
        ["字母 a 重复", "6.2%", "1.50", "26.2"],
    ],
    [250, 82, 88, 96],
    title="表 2  不同 prompt 的 MTP 接受率",
    note="共 5 组，同一模型同一 draft 深度，仅改变输出内容形态。",
    highlight_col=1,
)

draw_table(
    "t3-cost.png",
    ["场景", "每 token 位置成本 c"],
    [
        ["单流 fused（draft 8）", "0.72 ms"],
        ["单流 materialized（draft 15）", "1.08 ms"],
        ["8 路 batch materialized（draft 15）", "0.47 ms"],
    ],
    [270, 150],
    title="表 3  每轮耗时的边际成本",
    note="由 T = 32ms + c × 位置数 拟合；batch 越大固定开销摊得越薄。",
    highlight_col=1,
)

draw_table(
    "t4-spec.png",
    ["项目", "参数"],
    [
        ["GPU", "RTX 5090 Laptop，24 GB GDDR7，896 GB/s"],
        ["功耗", "默认 95 W，上限 175 W，实测峰值 179.86 W"],
        ["SM 频率 / 温度", "2377 MHz / 73 摄氏度"],
        ["引擎 / 模型", "ninfer-serve（e8-overlay-adaptive 分支）"],
        ["模型权重", "Qwen3.8-27B NVFP4，16.78 GiB"],
        ["KV 量化", "rk2v4-e8"],
    ],
    [130, 420],
    title="表 4  测试环境",
)


# ---------- 代码 ----------

draw_code(
    "c1-peak.png",
    [
        "[2026-11-05 14:22:43.298] ninfer-serve: throughput",
        "  interval=5.008s prefill=0.0tok/s decode=1508.0tok/s",
        "  running=8 prefilling=0 decode_ready=8 waiting=0",
        "  avg_decode_batch=8.00",
    ],
    title="代码 1  峰值采样原文（serve.log）",
    highlight_lines=[1],
)

draw_code(
    "c2-source.png",
    [
        "// src/targets/qwen3_6/impl/runtime/text_context_impl.h:685",
        "if (width > kDFlashDecodeMaximumWidth) {",
        "    // kDFlashDecodeMaximumWidth = 15 + 1 = 16",
        "    throw std::invalid_argument(",
        "        \"target verify batch shape is outside the supported domain\");",
        "}",
    ],
    title="代码 2  真正的 draft 上限",
    highlight_lines=[1],
)

draw_code(
    "c3-patch.png",
    [
        "1  src/ops/wrapper/mtp_round.cpp          T > 9      ->  T > 17",
        "2  targets/qwen3_6_27b/impl/config.h      Draft 8    ->  Draft 16",
        "3  src/product/speculative_options.h      > 8        ->  > 16",
        "4  targets/qwen3_6/.../round_state.h      Drafts 8   ->  Drafts 16",
        "5  src/ops/wrapper/gdn_input_proj.cpp     Width 16   ->  Width 17",
    ],
    title="代码 3  需要放开的 6 处限制（第 5 项含 4 处）",
    highlight_lines=[0, 4],
)

draw_code(
    "c4-run.png",
    [
        "# 1. 编译并应用补丁",
        "python patches/enable-draft15.py /path/to/ninfer",
        "python build_ninfer.py build",
        "",
        "# 2. 启动 8 路挡，草稿深度 15",
        "NINFER_DRAFT=15 NINFER_CUDA_GRAPH=1 \\",
        "    python build_ninfer.py serve8",
        "",
        "# 3. 压测",
        "python scripts/bench_quick.py 20 8 predict",
    ],
    title="代码 4  复现命令",
    highlight_lines=[5],
)

draw_table(
    "t5-error.png",
    ["曾对外说", "实际含义", "真实值"],
    [
        ["8 路 1200 / 1269 tok/s", "总 token 除标称 12 秒，实际跑了 29~103 秒", "489.6"],
        ["8 路 780", "引擎 5 秒窗口峰值，被当成持续吞吐", "约 485"],
    ],
    [210, 340, 90],
    title="表 5  我此前报错过的数字",
    note="两个口径错误：分母用标称时长；瞬时值当平均值。",
)

draw_code(
    "c5-stoplog.png",
    [
        "[req 614] done finish=stop_token gen=2 cache=40",
        "  reuse=restore_response_checkpoint",
    ],
    title="代码 5  缓存复用把请求秒杀的日志",
    highlight_lines=[0],
)

print("done")
