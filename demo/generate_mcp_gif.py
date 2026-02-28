#!/usr/bin/env python3
"""
Generate Terradev MCP Server demo GIF purely with Python + Pillow.
Shows Claude Code MCP interactions: GPU quotes, Wide-EP MoE deployment,
NIXL KV transfer, EP group health checks.

Commands are typed word-by-word. Output streams line-by-line.

Usage:
    python demo/generate_mcp_gif.py
    # Output: demo/terradev-mcp-demo.gif
"""

import os
from PIL import Image, ImageDraw, ImageFont

# ── Configuration ────────────────────────────────────────────────────────

CHAR_W = 10        # px per character (monospace)
CHAR_H = 20        # px per line
PAD_X = 24         # left/right padding
PAD_Y = 20         # top/bottom padding
COLS = 96          # terminal width in chars
MAX_LINES = 32     # fixed terminal height so all frames are the same size
BG = (30, 30, 46)  # Catppuccin Mocha background
FG = (205, 214, 244)
GREEN = (166, 227, 161)
YELLOW = (249, 226, 175)
RED = (243, 139, 168)
CYAN = (137, 220, 235)
BLUE = (137, 180, 250)
MAGENTA = (203, 166, 247)
DIM = (108, 112, 134)
ORANGE = (250, 179, 135)
TITLE_BAR = (24, 24, 37)
CURSOR_COLOR = (205, 214, 244)
PURPLE = (180, 142, 255)

IMG_W = COLS * CHAR_W + PAD_X * 2
IMG_H = MAX_LINES * CHAR_H + PAD_Y * 2 + 32

# Try to find a monospace font
FONT = None
FONT_PATHS = [
    "/System/Library/Fonts/SFMono-Regular.otf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.dfont",
    "/Library/Fonts/SF-Mono-Regular.otf",
]
for fp in FONT_PATHS:
    if os.path.exists(fp):
        try:
            FONT = ImageFont.truetype(fp, 14)
            break
        except Exception:
            continue
if FONT is None:
    FONT = ImageFont.load_default()


# ── Frame Renderer ───────────────────────────────────────────────────────

def render_frame(lines, title="Claude Code — terradev-mcp", cursor_pos=None):
    """
    Render a terminal frame at fixed size.
    lines: list of items, each either:
      - (text, color)            — single-color line
      - [(text, color), ...]     — multi-segment line
    cursor_pos: (line_idx, char_offset) to draw a block cursor, or None.
    """
    img = Image.new("RGB", (IMG_W, IMG_H), BG)
    draw = ImageDraw.Draw(img)

    # Title bar
    draw.rectangle([0, 0, IMG_W, 30], fill=TITLE_BAR)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([12 + i * 22, 9, 24 + i * 22, 21], fill=c)
    if title:
        draw.text((IMG_W // 2 - len(title) * 3, 7), title, fill=DIM, font=FONT)

    y_off = 32 + PAD_Y
    for line in lines:
        x = PAD_X
        if isinstance(line, tuple):
            text, color = line
            draw.text((x, y_off), text, fill=color, font=FONT)
        elif isinstance(line, list):
            for text, color in line:
                draw.text((x, y_off), text, fill=color, font=FONT)
                x += len(text) * CHAR_W
        y_off += CHAR_H

    # Draw block cursor
    if cursor_pos is not None:
        cy, cx = cursor_pos
        cx_px = PAD_X + cx * CHAR_W
        cy_px = 32 + PAD_Y + cy * CHAR_H
        draw.rectangle(
            [cx_px, cy_px, cx_px + CHAR_W, cy_px + CHAR_H],
            fill=CURSOR_COLOR,
        )

    return img


# ── Typing helpers ───────────────────────────────────────────────────────

def type_command(existing_lines, prompt_segs, cmd_text, word_delay=120):
    """
    Generate frames that type out `cmd_text` word-by-word after the prompt.
    Returns (frames, durations, final_lines).
    """
    frames = []
    durations = []
    words = cmd_text.split(" ")

    typed_so_far = ""
    prompt_char_len = sum(len(t) for t, _ in prompt_segs)

    for i, word in enumerate(words):
        if i > 0:
            typed_so_far += " "
        typed_so_far += word

        line = list(prompt_segs) + [(typed_so_far, FG)]
        cur_lines = existing_lines + [line]
        cursor_col = prompt_char_len + len(typed_so_far)
        frames.append(render_frame(cur_lines, cursor_pos=(len(cur_lines) - 1, cursor_col)))
        durations.append(word_delay)

    # Final hold with cursor
    final_line = list(prompt_segs) + [(cmd_text, FG)]
    final_lines = existing_lines + [final_line]
    frames.append(render_frame(final_lines, cursor_pos=(len(final_lines) - 1, prompt_char_len + len(cmd_text))))
    durations.append(400)

    return frames, durations, final_lines


def stream_output(existing_lines, output_lines, line_delay=80):
    """
    Generate frames that reveal output_lines one line at a time.
    Returns (frames, durations, final_lines).
    """
    frames = []
    durations = []
    cur = list(existing_lines)

    for line in output_lines:
        cur = cur + [line]
        frames.append(render_frame(cur))
        durations.append(line_delay)

    return frames, durations, cur


def hold(lines, ms):
    """Single hold frame."""
    return [render_frame(lines)], [ms]


# ── Demo Scenes ──────────────────────────────────────────────────────────

def build_frames():
    all_frames = []
    all_durations = []

    def add(f, d):
        all_frames.extend(f)
        all_durations.extend(d)

    prompt = [("claude ", PURPLE), ("> ", DIM)]

    # ═══════════════════════════════════════════════════════════════════
    # Scene 1: MCP Tool — quote_gpu H100
    # ═══════════════════════════════════════════════════════════════════

    f, d, lines = type_command([], prompt, "quote_gpu H100", word_delay=140)
    add(f, d)

    output1 = [
        ("", FG),
        [("MCP ", PURPLE), ("tool: quote_gpu", CYAN)],
        [("Scanning 15 providers in parallel...", DIM)],
    ]
    f, d, lines = stream_output(lines, output1, line_delay=500)
    add(f, d)
    add(*hold(lines, 800))

    header = "Provider        GPU    $/hr     Region        Spot"
    sep = "-" * len(header)
    table_lines = [
        ("", FG),
        (header, YELLOW),
        (sep, DIM),
        ("RunPod          H100   $2.49    US-TX         yes", GREEN),
        ("Vast.ai         H100   $2.69    US-Central    yes", GREEN),
        ("CoreWeave       H100   $3.04    LAS1          yes", FG),
        ("Lambda          H100   $3.29    us-west-1     no ", FG),
        ("AWS             H100   $4.68    us-east-1     yes", DIM),
        (sep, DIM),
        ("", FG),
        [("Best: ", FG), ("$2.49/hr RunPod", GREEN), (" — saves 51% vs AWS", YELLOW)],
    ]
    f, d, lines = stream_output(lines, table_lines, line_delay=70)
    add(f, d)
    add(*hold(lines, 2500))

    # ═══════════════════════════════════════════════════════════════════
    # Scene 2: MCP Tool — deploy_wide_ep (MoE Expert Parallelism)
    # ═══════════════════════════════════════════════════════════════════

    f, d, lines = type_command([], prompt, "deploy_wide_ep GLM-5-FP8 --dp 8 --ep", word_delay=120)
    add(f, d)

    ep_output = [
        ("", FG),
        [("MCP ", PURPLE), ("tool: deploy_wide_ep", CYAN)],
        ("", FG),
        [("Model:  ", DIM), ("zai-org/GLM-5-FP8", FG), ("  (744B params, 40B active)", DIM)],
        [("Layout: ", DIM), ("TP=1  DP=8  EP=8", YELLOW)],
        [("Flags:  ", DIM), ("--enable-expert-parallel --enable-eplb --enable-dbo", CYAN)],
        ("", FG),
        [("Building Ray Serve LLM deployment...", DIM)],
    ]
    f, d, lines = stream_output(lines, ep_output, line_delay=100)
    add(f, d)
    add(*hold(lines, 1000))

    ep_result = [
        ("", FG),
        [("EP Group ", MAGENTA), ("across 8 GPUs:", FG)],
        ("  Rank 0: experts   0-31   [GPU 0] NVLink domain A", FG),
        ("  Rank 1: experts  32-63   [GPU 1] NVLink domain A", FG),
        ("  Rank 2: experts  64-95   [GPU 2] NVLink domain A", FG),
        ("  Rank 3: experts  96-127  [GPU 3] NVLink domain A", FG),
        ("  Rank 4: experts 128-159  [GPU 4] NVLink domain B", FG),
        ("  Rank 5: experts 160-191  [GPU 5] NVLink domain B", FG),
        ("  Rank 6: experts 192-223  [GPU 6] NVLink domain B", FG),
        ("  Rank 7: experts 224-255  [GPU 7] NVLink domain B", FG),
        ("", FG),
        [("Env: ", DIM), ("VLLM_USE_DEEP_GEMM=1", CYAN)],
        [("      ", DIM), ("VLLM_ALL2ALL_BACKEND=deepep_low_latency", CYAN)],
        ("", FG),
        [("Deployed! ", GREEN), ("http://10.0.1.100:8000/v1", BLUE)],
    ]
    f, d, lines = stream_output(lines, ep_result, line_delay=65)
    add(f, d)
    add(*hold(lines, 3000))

    # ═══════════════════════════════════════════════════════════════════
    # Scene 3: MCP Tool — deploy_pd (NIXL KV Transfer)
    # ═══════════════════════════════════════════════════════════════════

    f, d, lines = type_command([], prompt, "deploy_pd GLM-5-FP8 --nixl --prefill-tp 8", word_delay=120)
    add(f, d)

    pd_output = [
        ("", FG),
        [("MCP ", PURPLE), ("tool: deploy_pd", CYAN)],
        ("", FG),
        [("Disaggregated Prefill/Decode ", MAGENTA), ("with NIXL KV transfer", FG)],
        ("", FG),
        [("  Prefill:  ", DIM), ("8 GPUs  TP=8", YELLOW), ("  (compute-bound)", DIM)],
        [("  Decode:   ", DIM), ("24 GPUs TP=1 DP=24", YELLOW), ("  (memory-bound)", DIM)],
        ("", FG),
        [("  KV Connector: ", DIM), ("NixlConnector", GREEN)],
        [("  Transport:    ", DIM), ("RDMA (mlx5_0)", CYAN), ("  zero-copy GPU-GPU", DIM)],
        [("  Buffer:       ", DIM), ("5 GB per endpoint", FG)],
        [("  Transfer:     ", DIM), ("~0.5ms / 1GB KV cache", GREEN)],
        ("", FG),
        [("P/D pair established ", GREEN), ("(NIXL+RDMA active)", CYAN)],
    ]
    f, d, lines = stream_output(lines, pd_output, line_delay=80)
    add(f, d)
    add(*hold(lines, 3000))

    # ═══════════════════════════════════════════════════════════════════
    # Scene 4: Closing tagline
    # ═══════════════════════════════════════════════════════════════════

    tag_lines = [
        ("", FG),
        ("", FG),
        ("", FG),
        ("", FG),
        [("   terradev-mcp", CYAN), (" — GPU provisioning for Claude Code", FG)],
        ("", FG),
        [("   ", FG), ("npm install -g terradev-mcp", GREEN)],
        ("", FG),
        [("   Ray Serve LLM", BLUE), (" . ", DIM), ("Expert Parallelism", MAGENTA),
         (" . ", DIM), ("NIXL KV transfer", ORANGE)],
        [("   EPLB", CYAN), (" . ", DIM), ("DeepEP/DeepGEMM", YELLOW),
         (" . ", DIM), ("15 cloud providers", GREEN)],
        ("", FG),
        [("   pip install terradev-cli", DIM), ("    ", FG),
         ("v3.3.0", GREEN)],
        [("   npm install -g terradev-mcp", DIM), (" ", FG),
         ("v1.5.0", GREEN)],
        ("", FG),
        [("   github.com/theoddden/terradev-mcp", DIM)],
        ("", FG),
    ]
    f, d, _ = stream_output([], tag_lines, line_delay=120)
    add(f, d)
    add(*hold(tag_lines, 3500))

    return all_frames, all_durations


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print("Generating MCP demo frames...")
    frames, durations = build_frames()
    print(f"  {len(frames)} frames built")

    out_path = os.path.join(out_dir, "terradev-mcp-demo.gif")
    print(f"Saving GIF to {out_path}...")

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Done! {out_path} ({size_kb:.0f} KB)")

    if size_kb > 5000:
        print("Warning: GIF is >5MB — consider reducing frame count or resolution")
    else:
        print("Size is under 5MB — good for GitHub/NPM README")


if __name__ == "__main__":
    main()
