"""Video input preparation and greedy generation.

Evaluation protocol of the paper: T frames sampled uniformly over the whole
clip (repeats allowed on short clips), each frame requested at about 114,688
pixels (56 visual tokens). The Qwen3-VL video processor additionally caps the
total pixels of a video (`size.longest_edge`, ~24 Mpx), so at T = 256 frames
are resized further, to about 39 visual tokens each; every number in the
paper was measured this way and the cap is deliberately left in place.
"""

from __future__ import annotations

from typing import List

import torch

from .modeling import merge_updates, unmerge_updates

PIXELS_PER_FRAME = 114688
_patched = False


def _install_video_reader():
    """Exact frame counts and a single-threaded decord reader.

    qwen_vl_utils rejects `nframes` larger than the clip's frame count; we
    allow it (frames repeat). decord's threaded decoder occasionally crashes
    on LLaVA-Video files, so it runs with one thread (same frames).
    """
    global _patched
    if _patched:
        return
    from qwen_vl_utils import vision_process as vp

    orig_nframes = vp.smart_nframes

    def smart_nframes(ele, total_frames, video_fps):
        if "nframes" in ele:
            return vp.round_by_factor(ele["nframes"], vp.FRAME_FACTOR)
        return orig_nframes(ele, total_frames=total_frames, video_fps=video_fps)

    def read_video_decord(ele):
        import decord
        vr = decord.VideoReader(ele["video"], num_threads=1)
        total, fps = len(vr), vr.get_avg_fps()
        start, end, total = vp.calculate_video_frame_range(ele, total, fps)
        n = vp.smart_nframes(ele, total_frames=total, video_fps=fps)
        idx = torch.linspace(start, end, n).round().long().tolist()
        video = torch.tensor(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
        meta = dict(fps=fps, frames_indices=idx, total_num_frames=total,
                    video_backend="decord")
        return video, meta, n / max(total, 1e-6) * fps

    vp.smart_nframes = smart_nframes
    vp._read_video_decord = read_video_decord
    vp.VIDEO_READER_BACKENDS["decord"] = read_video_decord
    _patched = True


def video_message(video_path: str, prompt: str, nframes: int = 256,
                  pixels: int = PIXELS_PER_FRAME) -> List[dict]:
    return [{"role": "user", "content": [
        {"type": "video", "video": video_path, "nframes": nframes,
         "min_pixels": int(pixels * 0.9), "max_pixels": pixels,
         "total_pixels": 10 ** 12},
        {"type": "text", "text": prompt}]}]


def prepare_inputs(processor, messages: List[dict], device="cuda"):
    """Chat template + frame decoding + processor -> model inputs."""
    _install_video_reader()
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, image_patch_size=16)
    video_kwargs = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
                    for k, v in video_kwargs.items()}
    return processor(text=[text], images=images, videos=videos,
                     return_tensors="pt", **video_kwargs).to(device)


@torch.no_grad()
def generate(model, processor, inputs, max_new_tokens: int = 128,
             merge: bool = True) -> str:
    """Greedy decoding, batch size 1.

    The KV cache is shorter than `input_ids` after contraction, so this runs
    its own prefill + decode loop instead of `model.generate`. With
    `merge=True` the generated updates are merged into the weights after
    prefill and removed afterwards (Sec. 3.5).
    """
    eos = processor.tokenizer.eos_token_id
    if not hasattr(model, "wire_state"):          # stock backbone
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, eos_token_id=eos)
        return processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()
    out = model(**inputs, use_cache=True, logits_to_keep=1)
    next_id = out.logits[:, -1].argmax(-1, keepdim=True)
    past = out.past_key_values
    ids = []
    if merge:
        merge_updates(model)
    try:
        for _ in range(max_new_tokens):
            tok = next_id.item()
            if tok == eos:
                break
            ids.append(tok)
            step = model(input_ids=next_id, past_key_values=past,
                         use_cache=True)
            next_id = step.logits[:, -1].argmax(-1, keepdim=True)
            past = step.past_key_values
    finally:
        if merge:
            unmerge_updates(model)
    return processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
