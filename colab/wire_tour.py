"""Helpers of the WIRE Colab tour (colab/wire_tour.ipynb).

The tour reads a real video with the Qwen3-VL processor and the repository's input preparation
(wire/inference.py), contracts it with the repository's rule (wire/compressor.py) and, on a GPU with enough
memory, runs the released model. This file adds the tour's figures, its animation and the comparison table.
"""
import io
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from wire.compressor import _aspect_factor, contraction_segments, frame_budgets, video_frame_runs  # noqa: E402
from wire.config import WireConfig  # noqa: E402
from wire.inference import _install_video_reader, prepare_inputs, video_message  # noqa: E402

SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/d/dc/Peacock_walking_and_eating.webm"
SAMPLE_CREDIT = "\"Peacock walking and eating\" by Mx. Granger, Wikimedia Commons, CC0 1.0"
CFG = WireConfig()
# Key-value cache of Qwen3-VL-8B per position: 36 layers, a key and a value, 8 key-value heads of width 128, bfloat16
KV_BYTES = 36 * 2 * 8 * 128 * 2
BLUE, AMBER, GRAY, LIGHT, TEXT, TOKEN, INK = "#2548F6", "#E8862A", "#6A6A6A", "#ABABAB", "#E9E3FB", "#D6D6D6", "#21222C"


# ---------------------------------------------------------------- the video
def fetch_video(src=""):
    """A local path for the sample clip (downloaded once), a URL, or a path of your own."""
    if src and not src.startswith(("http://", "https://")):
        return src
    url = src or SAMPLE_URL
    dest = os.path.join(HERE, "sample_clip.webm" if not src else "my_video" + os.path.splitext(url.split("?")[0])[1])
    if not os.path.exists(dest):
        req = urllib.request.Request(url, headers={"User-Agent": "WIRE-Colab-tour/1.0"})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


def _frame_indices(total, n):
    """The frame indices of the repository's reader: n frames sampled uniformly over the whole clip."""
    return torch.linspace(0, total - 1, n).round().long().tolist()


def _use_pyav_without_decord():
    """The repository reads videos with decord. Where decord has no wheel (such as macOS on Apple silicon), read
    the same frames with PyAV."""
    try:
        import decord  # noqa: F401
        return
    except ImportError:
        pass
    import av
    from qwen_vl_utils import vision_process as vp

    def read(ele):
        with av.open(ele["video"]) as c:
            st = c.streams.video[0]
            total = st.frames or sum(1 for _ in c.decode(st))
            fps = float(st.average_rate)
        n = vp.smart_nframes(ele, total_frames=total, video_fps=fps)
        idx = _frame_indices(total, n)
        want, frames = set(idx), {}
        with av.open(ele["video"]) as c:
            for i, fr in enumerate(c.decode(c.streams.video[0])):
                if i in want:
                    frames[i] = fr.to_ndarray(format="rgb24")
        video = torch.tensor(np.stack([frames[i] for i in idx])).permute(0, 3, 1, 2)
        return video, dict(fps=fps, frames_indices=idx, total_num_frames=total, video_backend="decord"), n / total * fps

    vp.VIDEO_READER_BACKENDS["decord"] = read
    vp.get_video_reader_backend = lambda: "decord"


def _thumbnails(path, n, width=320):
    """Small copies of the sampled frames, for the figures."""
    from PIL import Image
    try:
        import decord
        vr = decord.VideoReader(path, num_threads=1)
        return [Image.fromarray(a).resize((width, round(width * a.shape[0] / a.shape[1])))
                for a in vr.get_batch(_frame_indices(len(vr), n)).asnumpy()]
    except ImportError:
        import av
        with av.open(path) as c:
            frames = [fr.to_image() for fr in c.decode(c.streams.video[0])]
        return [frames[i].resize((width, round(width * frames[i].height / frames[i].width)))
                for i in _frame_indices(len(frames), n)]


def load_processor():
    """The Qwen3-VL processor of the backbone (tokenizer and video processor, about 15 MB)."""
    import warnings
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore", message="IProgress not found")
    from huggingface_hub.utils import logging as hub_logging
    from transformers import AutoProcessor
    from transformers.utils import logging
    logging.set_verbosity_error()
    hub_logging.set_verbosity_error()
    return AutoProcessor.from_pretrained(CFG.base_model)


@dataclass
class Clip:
    inputs: dict           # model inputs from wire.inference.prepare_inputs
    frames: int            # sampled frames
    pairs: int             # frame pairs: Qwen3-VL turns two consecutive frames into one token grid
    grid: tuple            # token grid of one pair, rows by columns
    per_pair: int          # visual tokens per pair
    kept_per_pair: int     # refined tokens per pair, with 16 per frame (Eq. 3)
    full: int              # positions of the decoder input
    visual: int            # visual positions among them
    wire: int              # positions after contraction
    segments: list         # segment of each position of the first pair (Eq. 6)
    kept_positions: list   # positions of the first pair that the refined tokens take
    thumbs: list           # one small image per pair
    is_visual: list        # whether each position of the decoder input is visual


def read_clip(processor, video, question, frames=16, k=CFG.tokens_kept_per_frame):
    """Prepare the decoder input of WIRE for a video and a question, and contract it with the repository's rule."""
    _install_video_reader()
    _use_pyav_without_decord()
    x = prepare_inputs(processor, video_message(video, question, nframes=frames), "cpu")
    ids = x["input_ids"][0]
    is_vis = ids == processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vpos = is_vis.nonzero().flatten()
    run_id, run_len = video_frame_runs(vpos, CFG.tokens_per_frame)
    kt = frame_budgets(run_len, k, CFG.tokens_per_frame)
    seg, last = contraction_segments(run_id, run_len, kt)
    _, gh, gw = [int(v) for v in x["video_grid_thw"][0]]
    pairs, per = int(run_len.numel()), int(run_len[0])
    thumbs = _thumbnails(video, frames)[::2]
    return Clip(inputs=x, frames=frames, pairs=pairs, grid=(gh // 2, gw // 2), per_pair=per,
                kept_per_pair=int(kt[0]), full=ids.numel(), visual=vpos.numel(),
                wire=ids.numel() - vpos.numel() + int(kt.sum()), segments=seg[:per].tolist(),
                kept_positions=(vpos[last][run_id[last] == 0] - vpos[0]).tolist(), thumbs=thumbs, is_visual=is_vis.tolist())


def load_results(path=None):
    """The paper's measurements used by the tour (colab/paper_results.json)."""
    with open(path or os.path.join(HERE, "paper_results.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- figures
def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _grid(ax, img, gh, gw):
    from matplotlib.patches import Rectangle
    W, H = img.size
    ax.imshow(img)
    for i in range(gh):
        for j in range(gw):
            ax.add_patch(Rectangle((j * W / gw, i * H / gh), W / gw, H / gh, fill=False, ec="white", lw=0.5, alpha=0.7))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _tokens(ax, n, gh, gw, title=None):
    """A block of n refined tokens, laid out as the compressor lays out its queries for this frame shape."""
    from matplotlib.patches import Rectangle
    a, b = _aspect_factor(n, gh, gw)
    ax.set_xlim(0, b); ax.set_ylim(a, 0); ax.set_aspect("equal"); ax.axis("off")
    for i in range(a):
        for j in range(b):
            ax.add_patch(Rectangle((j + 0.08, i + 0.08), 0.84, 0.84, fc=AMBER, ec="none"))
    if title:
        ax.set_title(title, fontsize=10, color=INK)


def show_input(clip):
    """Figure: the sampled frames as token grids, and the composition of the decoder input."""
    plt = _style()
    from matplotlib.patches import Rectangle
    shown = np.linspace(0, clip.pairs - 1, min(8, clip.pairs)).round().astype(int).tolist()
    fig = plt.figure(figsize=(12, 2.3))
    gh, gw = clip.grid
    for slot, k in enumerate(shown):
        ax = fig.add_axes([0.005 + slot * 0.124, 0.4, 0.118, 0.48])
        _grid(ax, clip.thumbs[k], gh, gw)
        ax.set_title(f"frames {2 * k + 1} and {2 * k + 2}", fontsize=9, color=GRAY)
    vis = clip.is_visual
    ax = fig.add_axes([0.005, 0.15, 0.99, 0.14]); ax.set_xlim(0, clip.full); ax.set_ylim(0, 1); ax.axis("off")
    start = 0
    for p in range(1, clip.full + 1):
        if p == clip.full or vis[p] != vis[start]:
            ax.add_patch(Rectangle((start, 0.1), p - start, 0.8, fc=LIGHT if vis[start] else TEXT, ec="white", lw=0.4))
            start = p
    fig.text(0.005, 0.0, f"Decoder input: {clip.full:,} positions, {clip.visual:,} visual (grey) and {clip.full - clip.visual:,} text "
             "(lilac: prompt, timestamps and question)", fontsize=10, color=INK)
    return fig


def show_refine(clip):
    """Figure: one frame pair becomes a few refined tokens, which take the last position of each segment."""
    plt = _style()
    from matplotlib.patches import Rectangle
    gh, gw = clip.grid
    per, kept = clip.per_pair, clip.kept_per_pair
    fig = plt.figure(figsize=(12, 3.6))
    ax = fig.add_axes([0.005, 0.2, 0.2, 0.62]); _grid(ax, clip.thumbs[0], gh, gw)
    ax.set_title(f"{per} visual tokens", fontsize=10, color=INK)
    fig.text(0.215, 0.53, "→", fontsize=26, color=GRAY, va="center")
    fig.text(0.212, 0.42, "scan", fontsize=9, color=GRAY)
    _tokens(fig.add_axes([0.255, 0.28, 0.14, 0.45]), kept, gh, gw, f"{kept} refined tokens")
    ax = fig.add_axes([0.43, 0.6, 0.56, 0.22]); ax.set_xlim(0, per); ax.set_ylim(0, 1)
    segs, keep = clip.segments, set(clip.kept_positions)
    for q in range(per):
        if q == 0 or segs[q] != segs[q - 1]:
            end = q
            while end < per and segs[end] == segs[q]:
                end += 1
            ax.add_patch(Rectangle((q, 0), end - q, 1, fc="#F1F1F1" if segs[q] % 2 else "#E4E4E4", ec="none"))
        ax.add_patch(Rectangle((q + 0.12, 0.15), 0.76, 0.7, fc=AMBER if q in keep else TOKEN, ec="none"))
    ax.set_yticks([]); ax.set_xticks([0, per - 1]); ax.set_xticklabels(["0", str(per - 1)], fontsize=8.5)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"One frame pair: {per} positions in {kept} segments; a refined token takes the last of each",
                 fontsize=10.5, loc="left", color=INK)
    ax = fig.add_axes([0.43, 0.08, 0.56, 0.3]); ax.set_xlim(0, clip.full * 1.5); ax.set_ylim(-0.6, 1.6); ax.axis("off")
    for y, (lab, n, col) in zip([1, 0], [("Full input", clip.full, LIGHT), ("WIRE", clip.wire, AMBER)]):
        ax.add_patch(Rectangle((0, y - 0.35), n, 0.7, fc=col, ec="none"))
        ax.text(n + clip.full * 0.015, y, f"{lab}: {n:,} positions", va="center", fontsize=10, color=INK)
    ax.set_title(f"The whole input: {clip.full / clip.wire:.1f}× shorter", fontsize=10.5, loc="left", color=INK)
    return fig


def show_savings(clip, results):
    """Figure: cache size for this clip and at the paper's 256 frames."""
    plt = _style()
    t4 = [r for r in results["table_4_latency_ms"]["rows"] if r["k"] == 16 and r["frames"] == 256][0]
    p3 = results["setup_section_3"]["positions_at_256_frames"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.0))
    for ax, (title, full, wire, unit, div) in zip(axes, [
            (f"Cache size for this clip, {clip.frames} frames", clip.full, clip.wire, "MB", 2 ** 20),
            ("Cache size at the paper's 256 frames", p3["total"], 5154, "GB", 2 ** 30)]):
        a, b = full * KV_BYTES / div, wire * KV_BYTES / div
        ax.barh([1, 0], [a, b], color=[LIGHT, AMBER], height=0.55)
        for y, v in zip([1, 0], [a, b]):
            ax.text(v + a * 0.02, y, f"{v:,.0f} {unit}" if unit == "MB" else f"{v:.2f} {unit}", va="center", fontsize=10)
        ax.set_yticks([1, 0]); ax.set_yticklabels(["Uncompressed backbone", "WIRE"]); ax.set_xticks([])
        ax.set_xlim(0, a * 1.25); ax.spines["bottom"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    fig.text(0.005, 0.0, "Cache sizes follow from the positions and the dimensions of Qwen3-VL-8B. Measured at 256 frames, prefill "
             f"takes {t4['prefill_wire']} ms instead of {t4['prefill_base']} ms (Table 4).", fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def pair_animation(clip, max_pairs=16, duration=600):
    """An animation: pair after pair, the input of the backbone and the input of WIRE grow. Returns GIF bytes."""
    plt = _style()
    from matplotlib.patches import Rectangle
    from PIL import Image
    gh, gw = clip.grid
    order = np.linspace(0, clip.pairs - 1, min(clip.pairs, max_pairs)).round().astype(int).tolist()
    top = clip.pairs * clip.per_pair
    images = []
    for step, k in enumerate(order, 1):
        fig = plt.figure(figsize=(9.6, 2.8), dpi=80)
        ax = fig.add_axes([0.01, 0.12, 0.24, 0.66]); _grid(ax, clip.thumbs[k], gh, gw)
        ax.set_title(f"Frame pair {k + 1} of {clip.pairs}", fontsize=10, color=INK)
        fig.text(0.262, 0.47, "→", fontsize=22, color=GRAY, va="center")
        _tokens(fig.add_axes([0.3, 0.2, 0.13, 0.5]), clip.kept_per_pair, gh, gw, f"{clip.kept_per_pair} refined tokens")
        ax = fig.add_axes([0.47, 0.2, 0.52, 0.55]); ax.set_xlim(0, top * 1.35); ax.set_ylim(-0.6, 1.6); ax.axis("off")
        done = k + 1
        for y, (lab, n, col) in zip([1, 0], [("Backbone", done * clip.per_pair, LIGHT), ("WIRE", done * clip.kept_per_pair, AMBER)]):
            ax.add_patch(Rectangle((0, y - 0.35), n, 0.7, fc=col, ec="none"))
            ax.text(n + top * 0.015, y, f"{lab}: {n:,}", va="center", fontsize=10, color=INK)
        ax.set_title("Visual positions in the decoder input", fontsize=10, loc="left", color=INK)
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        buf.seek(0); images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    out = io.BytesIO()
    images[0].save(out, format="GIF", save_all=True, append_images=images[1:] + [images[-1]] * 4, duration=duration, loop=0)
    return out.getvalue()


# ---------------------------------------------------------------- asking the model
def gpu_ready(min_gb=22):
    """(ready, description): the released model needs a GPU with about 22 GB free for WIRE and the backbone."""
    if not torch.cuda.is_available():
        return False, "no GPU"
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 2 ** 30
    return gb >= min_gb, f"{p.name}, {gb:.0f} GB"


def ask(clip, processor, model_name="wire-k16", max_new_tokens=64):
    """Ask the question three ways on the GPU: the uncompressed backbone, WIRE, and WIRE without its update.
    Returns one row per system with the answer, the time to first token and the decoder positions."""
    from wire import controls
    from wire.inference import generate
    from wire.models import load
    model, _, _ = load(model_name, "cuda")
    inputs = clip.inputs.to("cuda")
    runs = [("Uncompressed backbone", {"teacher_mode": True}, False, clip.full),
            ("WIRE", {}, True, clip.wire),
            ("WIRE without its update", {"no_update": True}, True, clip.wire)]
    with torch.no_grad():
        model(**inputs, use_cache=True, logits_to_keep=1)          # warm-up
    rows = []
    for name, ctl, merge, positions in runs:
        with controls(model, **ctl):
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                model(**inputs, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize(); ttft = (time.perf_counter() - t) * 1000
            answer = generate(model, processor, inputs, max_new_tokens=max_new_tokens, merge=merge)
        rows.append(dict(system=name, answer=answer, ttft_ms=round(ttft), positions=positions))
    return rows


def show_answers(rows, question, note):
    """The comparison table as HTML."""
    from html import escape
    from IPython.display import HTML
    head = "".join(f"<th style='text-align:left;padding:6px 10px'>{h}</th>" for h in ("System", "Answer", "First token", "Positions"))
    body = ""
    for r in rows:
        bg = "#FFF4E8" if r["system"] == "WIRE" else "#F4F5F7"
        ttft = f"{r['ttft_ms']:,} ms" if r.get("ttft_ms") is not None else ""
        body += (f"<tr style='background:{bg}'><td style='padding:6px 10px'><b>{escape(r['system'])}</b></td>"
                 f"<td style='padding:6px 10px'>{escape(r.get('answer') or '')}</td><td style='padding:6px 10px'>{ttft}</td>"
                 f"<td style='padding:6px 10px'>{r['positions']:,}</td></tr>")
    return HTML(f"<p><b>Q:</b> {escape(question)}</p><table style='border-collapse:separate;border-spacing:0 4px'>"
                f"<tr>{head}</tr>{body}</table><p style='color:#6A6A6A;font-size:12px'>{escape(note)}</p>")
