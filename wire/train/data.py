"""Training data: LLaVA-Video-178K question-answer pairs.

Records are jsonl lines in the ms-swift format the paper's runs used:

    {"messages": [{"role": "user", "content": "<video>question"},
                  {"role": "assistant", "content": "answer"}],
     "videos": ["/abs/path/clip.mp4"]}

Frames are sampled at 2 fps (at most 128, at least 4) and every frame is
resized to 114,688 pixels (56 visual tokens). Unlike evaluation, the frame
timestamps in the prompt use the clip's real frame rate. The loss covers the
answer and the closing `<|im_end|>\n`, exactly as ms-swift's `qwen3_vl`
template labels it.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

import torch

PIXELS = 114688
# The paper's runs set qwen-vl-utils' VIDEO_MIN/MAX_TOKEN_NUM to 112 (through
# ms-swift), which makes the effective per-frame range
# [PIXELS, 1.05 * PIXELS]; the same range is passed explicitly here.
TRAIN_VIDEO = {"fps": 2.0, "max_frames": 128, "min_frames": 4,
               "min_pixels": PIXELS, "max_pixels": int(PIXELS * 1.05),
               "total_pixels": 128 * PIXELS}


def read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class VideoQADataset(torch.utils.data.Dataset):
    """Encodes one record per item. Records that fail to decode or exceed
    `max_length` tokens are replaced by a random other record (as ms-swift
    does with `truncation_strategy=delete`)."""

    def __init__(self, path: str, processor, max_length: int = 16384,
                 seed: int = 42, video_root: Optional[str] = None):
        self.records = read_jsonl(path)
        if video_root:                  # relative video paths in the jsonl
            for r in self.records:
                r["videos"] = [os.path.join(video_root, v) for v in r["videos"]]
        self.processor = processor
        self.max_length = max_length
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        for _ in range(100):
            try:
                enc = encode(self.processor, self.records[i])
                if enc["input_ids"].shape[0] <= self.max_length:
                    return enc
            except Exception as e:  # unreadable video
                print(f"[data] skipping record {i}: {type(e).__name__}: {e}",
                      flush=True)
            i = self.rng.randrange(len(self.records))
        raise RuntimeError("100 consecutive records failed to encode")


def encode(processor, record: dict) -> Dict[str, torch.Tensor]:
    from qwen_vl_utils import fetch_video

    user, assistant = record["messages"][0], record["messages"][1]
    question = user["content"].replace("<video>", "", 1)
    video, meta = fetch_video({"video": record["videos"][0], **TRAIN_VIDEO},
                              image_patch_size=16, return_video_metadata=True)
    messages = [{"role": "user", "content": [{"type": "video"},
                                              {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    out = processor(text=[prompt], videos=[video.to(torch.uint8)],
                    video_metadata=[meta], do_resize=False,
                    do_sample_frames=False, return_tensors="pt")
    answer = processor.tokenizer(assistant["content"] + "<|im_end|>\n",
                                 add_special_tokens=False,
                                 return_tensors="pt")["input_ids"][0]
    ids = torch.cat([out["input_ids"][0], answer])
    labels = torch.cat([torch.full_like(out["input_ids"][0], -100), answer])
    enc = {"input_ids": ids, "labels": labels,
           "pixel_values_videos": out["pixel_values_videos"],
           "video_grid_thw": out["video_grid_thw"]}
    if "mm_token_type_ids" in out:
        enc["mm_token_type_ids"] = torch.cat(
            [out["mm_token_type_ids"][0], torch.zeros_like(answer)])
    return enc


def collate(batch: List[dict], pad_id: int) -> Dict[str, torch.Tensor]:
    """Right padding; videos are concatenated along the patch dimension."""
    n = max(b["input_ids"].shape[0] for b in batch)

    def pad(key, value):
        return torch.stack([torch.cat([b[key], b[key].new_full(
            (n - b[key].shape[0],), value)]) for b in batch])

    out = {"input_ids": pad("input_ids", pad_id),
           "labels": pad("labels", -100),
           "attention_mask": torch.stack([torch.cat([
               torch.ones(b["input_ids"].shape[0], dtype=torch.long),
               torch.zeros(n - b["input_ids"].shape[0], dtype=torch.long)])
               for b in batch]),
           "pixel_values_videos": torch.cat([b["pixel_values_videos"] for b in batch]),
           "video_grid_thw": torch.cat([b["video_grid_thw"] for b in batch])}
    if all("mm_token_type_ids" in b for b in batch):
        out["mm_token_type_ids"] = pad("mm_token_type_ids", 0)
    return out


def build_collator(processor, pad_id: Optional[int] = None):
    pid = pad_id if pad_id is not None else processor.tokenizer.pad_token_id
    return lambda batch: collate(batch, pid)
