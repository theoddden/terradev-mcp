#!/usr/bin/env python3
"""
Generate Terradev MCP Server demo GIF purely with Python + Pillow.

All output formats match the actual codebase:
  - Scene 1: quote_gpu  → real CLI format from cli.py:1564-1571
  - Scene 2: moe_deploy → real MCP handler from terradev_mcp.py:1958-1977
  - Scene 3: infer_route_disagg → real MCP handler from terradev_mcp.py:1796-1822
  - Scene 4: provision_gpu → real CLI format from cli.py:2278-2294

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
# All output formats sourced from actual codebase:
#   cli.py:1564-1571        — quote table format
#   cli.py:2278-2294        — provision results format
#   terradev_mcp.py:1424-1451 — local_scan MCP output
#   terradev_mcp.py:1796-1822 — infer_route_disagg MCP output
#   terradev_mcp.py:1938-1982 — moe_deploy MCP output

def build_frames():
    all_frames = []
    all_durations = []

    def add(f, d):
        all_frames.extend(f)
        all_durations.extend(d)

    # Claude Code shows tool calls as: ToolName(args) then result
    # We simulate the Claude Code conversation view

    prompt = [("You: ", PURPLE)]

    # ═══════════════════════════════════════════════════════════════════
    # Scene 1: quote_gpu — exact CLI output format from cli.py:1564-1571
    # ═══════════════════════════════════════════════════════════════════

    f, d, lines = type_command([], prompt, "Get H100 prices across all providers", word_delay=110)
    add(f, d)

    # Claude Code shows tool call name then result
    tool_header = [
        ("", FG),
        [("Claude: ", BLUE), ("Using ", DIM), ("quote_gpu", CYAN), (" tool", DIM)],
        [("   quote ", FG), ("-g H100", YELLOW)],
        ("", FG),
        # Real output: cli.py line 1527
        ("Querying providers for H100 pricing...", DIM),
    ]
    f, d, lines = stream_output(lines, tool_header, line_delay=400)
    add(f, d)
    add(*hold(lines, 600))

    # Real quote table format: cli.py lines 1564-1571
    # print(f"\nTerradev Quote — {gpu_type}")
    # print(f"{'#':<4} {'Provider':<14} {'Region':<16} {'$/hr':<10} {'Type':<10}")
    # print("-" * 58)
    # print(f"{i+1:<4} {q['provider']:<14} {q['region']:<16} ${q['price']:<9.2f} {spot:<10}")
    table_lines = [
        ("", FG),
        ("Terradev Quote — H100", YELLOW),
        [("#   ", DIM), ("Provider      ", FG), ("Region           ", FG), ("$/hr      ", FG), ("Type", FG)],
        ("-" * 58, DIM),
        [("1   ", DIM), ("RunPod        ", FG), ("US-TX            ", FG), ("$2.49     ", GREEN), ("spot", GREEN)],
        [("2   ", DIM), ("Vast.ai       ", FG), ("US-Central       ", FG), ("$2.69     ", GREEN), ("spot", GREEN)],
        [("3   ", DIM), ("TensorDock    ", FG), ("US-East          ", FG), ("$2.89     ", FG), ("spot", FG)],
        [("4   ", DIM), ("CoreWeave     ", FG), ("LAS1             ", FG), ("$3.04     ", FG), ("spot", FG)],
        [("5   ", DIM), ("Lambda        ", FG), ("us-west-1        ", FG), ("$3.29     ", FG), ("on-demand", DIM)],
        [("6   ", DIM), ("AWS           ", FG), ("us-east-1        ", FG), ("$4.68     ", DIM), ("spot", DIM)],
        ("", FG),
        # Real output: cli.py lines 1571-1573
        [("Best: ", FG), ("$2.49/hr on RunPod (US-TX)", GREEN)],
        [("Estimated monthly: ", DIM), ("$1,818", YELLOW)],
        ("", FG),
        # Real output: cli.py lines 1587-1588
        [("Provision: ", DIM), ("terradev provision -g H100", CYAN)],
    ]
    f, d, lines = stream_output(lines, table_lines, line_delay=55)
    add(f, d)
    add(*hold(lines, 2800))

    # ═══════════════════════════════════════════════════════════════════
    # Scene 2: moe_deploy — exact MCP handler from terradev_mcp.py:1938-1977
    # ═══════════════════════════════════════════════════════════════════

    f, d, lines = type_command([], prompt, "Deploy GLM-5 MoE on 8x H100 with EP", word_delay=110)
    add(f, d)

    moe_header = [
        ("", FG),
        [("Claude: ", BLUE), ("Using ", DIM), ("moe_deploy", CYAN), (" tool", DIM)],
        [("   provision --task clusters/moe-template/task.yaml \\", FG)],
        [("     --set model_id=zai-org/GLM-5-FP8 --set tp_size=8", CYAN)],
        ("", FG),
    ]
    f, d, lines = stream_output(lines, moe_header, line_delay=200)
    add(f, d)

    # Real MCP output: terradev_mcp.py lines 1958-1963
    # output_text = f"🧬 **MoE Cluster Deployment**\n\n"
    # output_text += f"**Model:** {model_id}\n"
    # output_text += f"**GPU:** {gpu_type} × {tp_size} (TP={tp_size})\n"
    # output_text += f"**Backend:** {backend}\n"
    # output_text += f"**Quantization:** {quantization}\n"
    # output_text += f"**Dry Run:** {dry_run}\n\n"
    moe_result = [
        ("MoE Cluster Deployment", MAGENTA),
        ("", FG),
        [("Model:        ", DIM), ("zai-org/GLM-5-FP8", FG)],
        [("GPU:          ", DIM), ("H100 x 8 (TP=8)", YELLOW)],
        [("Backend:      ", DIM), ("vllm", FG)],
        [("Quantization: ", DIM), ("fp8", FG)],
        [("Dry Run:      ", DIM), ("False", FG)],
        ("", FG),
        # Real fallback output: terradev_mcp.py lines 1970-1977
        ("Manual deployment:", DIM),
        [("  terradev provision --task clusters/moe-template/task.yaml \\", CYAN)],
        [("    --set model_id=zai-org/GLM-5-FP8 --set tp_size=8", CYAN)],
        ("", FG),
        ("Or via Kubernetes:", DIM),
        [("  kubectl apply -f clusters/moe-template/k8s/", CYAN)],
        ("", FG),
        ("Or via Helm:", DIM),
        [("  helm upgrade --install moe-inf ./helm/terradev \\", CYAN)],
        [("    -f clusters/moe-template/helm/values-moe.yaml", CYAN)],
    ]
    f, d, lines = stream_output(lines, moe_result, line_delay=55)
    add(f, d)
    add(*hold(lines, 3000))

    # ═══════════════════════════════════════════════════════════════════
    # Scene 3: infer_route_disagg — exact MCP handler from
    # terradev_mcp.py:1796-1822
    # ═══════════════════════════════════════════════════════════════════

    f, d, lines = type_command([], prompt, "Route GLM-5 with disaggregated prefill/decode", word_delay=100)
    add(f, d)

    disagg_header = [
        ("", FG),
        [("Claude: ", BLUE), ("Using ", DIM), ("infer_route_disagg", CYAN), (" tool", DIM)],
        [("   inference route --disagg --model GLM-5 --check", FG)],
        ("", FG),
    ]
    f, d, lines = stream_output(lines, disagg_header, line_delay=200)
    add(f, d)

    # Real MCP output: terradev_mcp.py lines 1806-1817
    # output_text = "⚡ **Disaggregated Prefill/Decode Routing (DistServe)**\n\n"
    # output_text += f"**Model:** {arguments['model']}\n"
    # output_text += "**Architecture:** DistServe — PREFILL (compute-bound) → DECODE (memory-bound)\n"
    # output_text += "**KV Cache Handoff:** tracked via PrefillDecodeTracker\n\n"
    disagg_result = [
        ("Disaggregated Prefill/Decode Routing (DistServe)", MAGENTA),
        ("", FG),
        [("Model:            ", DIM), ("GLM-5", FG)],
        [("Architecture:     ", DIM), ("DistServe", YELLOW)],
        [("  PREFILL ", CYAN), ("(compute-bound)", DIM), (" -> ", FG),
         ("DECODE ", CYAN), ("(memory-bound)", DIM)],
        [("KV Cache Handoff: ", DIM), ("tracked via PrefillDecodeTracker", FG)],
        ("", FG),
        # Additional detail from inference_router.py KV connector
        [("KV Connector:     ", DIM), ("NixlConnector", GREEN)],
        [("Transport:        ", DIM), ("RDMA", CYAN), (" (rdma_device=mlx5_0)", DIM)],
        [("Buffer Size:      ", DIM), ("5 GB", FG)],
        ("", FG),
        ("PREFILL endpoints: high-FLOPS GPUs (H100 SXM)", DIM),
        ("DECODE endpoints:  high-bandwidth GPUs (H200, MI300X)", DIM),
    ]
    f, d, lines = stream_output(lines, disagg_result, line_delay=60)
    add(f, d)
    add(*hold(lines, 3000))

    # ═══════════════════════════════════════════════════════════════════
    # Scene 4: provision_gpu — exact CLI output from cli.py:2278-2294
    # ═══════════════════════════════════════════════════════════════════

    f, d, lines = type_command([], prompt, "Provision 4x H100 across clouds", word_delay=110)
    add(f, d)

    prov_header = [
        ("", FG),
        [("Claude: ", BLUE), ("Using ", DIM), ("provision_gpu", CYAN), (" tool", DIM)],
        [("   Terraform-powered parallel provisioning", FG)],
        ("", FG),
        # Real output: cli.py lines 1990-1991
        ("Provisioning 4x H100 (parallel=6)", FG),
        ("Querying all providers for real-time pricing...", DIM),
    ]
    f, d, lines = stream_output(lines, prov_header, line_delay=300)
    add(f, d)
    add(*hold(lines, 800))

    # Real output: cli.py lines 2031, 2098-2100
    prov_mid = [
        ("12 quotes — building allocation plan...", DIM),
        ("Deploying 4 instance(s) across 4 cloud(s) simultaneously...", FG),
        ("   RunPod / US-TX — $2.49/hr", DIM),
        ("   Vast.ai / US-Central — $2.69/hr", DIM),
        ("   CoreWeave / LAS1 — $3.04/hr", DIM),
        ("   Lambda / us-west-1 — $3.29/hr", DIM),
    ]
    f, d, lines = stream_output(lines, prov_mid, line_delay=100)
    add(f, d)
    add(*hold(lines, 1000))

    # Real output: cli.py lines 2278-2294
    # print(f"{'Provider':<14} {'Instance ID':<36} {'$/hr':<8} {'ms':<8}")
    # print("-" * 70)
    # print(f"{r['provider']:<14} {r['instance_id']:<36} ${r['price']:<7.2f} {r['elapsed_ms']:<.0f}ms")
    # print(f"\nTotal: ${total_hr:.2f}/hr  (${total_hr*24:.2f}/day)")
    # print(f"Group: {group_id}")
    # print(f"Total provision time: {provision_time:.0f}ms")
    prov_result = [
        ("", FG),
        ("=" * 60, DIM),
        ("4/4 instances launched across 4 cloud(s)", GREEN),
        [("Provider      ", FG), ("Instance ID                          ", FG), ("$/hr    ", FG), ("ms", FG)],
        ("-" * 70, DIM),
        [("RunPod        ", FG), ("rpd_1709142000_a7b3c1f2              ", FG), ("$2.49   ", GREEN), ("847ms", FG)],
        [("Vast.ai       ", FG), ("vst_1709142000_d4e5f6a8              ", FG), ("$2.69   ", GREEN), ("1203ms", FG)],
        [("CoreWeave     ", FG), ("cwv_1709142001_b8c9d0e1              ", FG), ("$3.04   ", FG), ("956ms", FG)],
        [("Lambda        ", FG), ("lbd_1709142001_f2a3b4c5              ", FG), ("$3.29   ", FG), ("1104ms", FG)],
        ("", FG),
        [("Total: ", FG), ("$11.51/hr", GREEN), ("  ($276.24/day)", DIM)],
        [("Group: ", DIM), ("pg_1709142000_a7b3c1f2", FG)],
        [("Total provision time: ", DIM), ("1847ms", GREEN)],
    ]
    f, d, lines = stream_output(lines, prov_result, line_delay=55)
    add(f, d)
    add(*hold(lines, 3000))

    # ═══════════════════════════════════════════════════════════════════
    # Scene 5: Closing tagline
    # ═══════════════════════════════════════════════════════════════════

    tag_lines = [
        ("", FG),
        ("", FG),
        ("", FG),
        [("   terradev-mcp", CYAN), (" — GPU provisioning for Claude Code", FG)],
        ("", FG),
        [("   ", FG), ("npm install -g terradev-mcp", GREEN)],
        [("   ", FG), ("pip install terradev-cli", GREEN)],
        ("", FG),
        [("   MoE Cluster Templates", BLUE), (" . ", DIM),
         ("Expert Parallelism", MAGENTA)],
        [("   NIXL KV Transfer", ORANGE), (" . ", DIM),
         ("DistServe P/D Routing", CYAN)],
        [("   Terraform Parallel Provisioning", YELLOW), (" . ", DIM),
         ("15 Clouds", GREEN)],
        ("", FG),
        [("   github.com/theoddden/terradev-mcp", DIM)],
        [("   pypi.org/project/terradev-cli", DIM), ("  v3.3.0", GREEN)],
        [("   npmjs.com/package/terradev-mcp", DIM), ("  v1.5.1", GREEN)],
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
