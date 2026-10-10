"""WIRE as an lmms-eval model (`--model wire`).

Same input protocol as eval/videomme/run.py: `nframes` frames sampled
uniformly over the clip, ~114,688 pixels per frame, greedy decoding with
WIRE's own prefill + decode loop (the contracted KV cache is shorter than
input_ids, so HF `generate` cannot be used).

model_args:
  model=wire-k16        registry name, checkpoint dir, Hub repo id,
                        "stock" or "pool<k>" (see wire.models.load)
  condition=own         own | no_update | zero_context | frame_drop
  nframes=256
  max_new_tokens=0      0 = the task's own value
"""

import os
import sys
from typing import List, Tuple

import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wire import controls  # noqa: E402
from wire.inference import generate, prepare_inputs, video_message  # noqa: E402
from wire.models import load  # noqa: E402

CONDITIONS = {
    "own": {},
    "no_update": {"no_update": True},
    "zero_context": {"zero_context": True},
    "frame_drop": {"frame_drop": 0.5, "frame_drop_seed": 0},
    "blind": {"zero_context": True, "no_update": True},
}
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@register_model("wire")
class Wire(lmms):
    def __init__(self, model: str = "wire-k16", condition: str = "own",
                 nframes: int = 256, max_new_tokens: int = 0,
                 device: str = "cuda:0", batch_size: int = 1, **kwargs) -> None:
        super().__init__()
        assert int(batch_size) == 1, "WIRE evaluates one question at a time"
        self._device = torch.device(device)
        self.nframes = int(nframes)
        self.force_max_new_tokens = int(max_new_tokens)
        self._model, self.processor, base = load(model, device)
        self._tokenizer = self.processor.tokenizer
        if base is None:
            if condition != "own":
                raise ValueError(f"condition {condition!r} needs a WIRE model")
            self.ctl = None
        else:
            self.ctl = {**CONDITIONS[condition], **base}
        self._rank, self._world_size = 0, 1
        self.batch_size_per_gpu = 1

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return 32768

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError

    def _answer(self, video: str, prompt: str, max_new_tokens: int) -> str:
        inputs = prepare_inputs(self.processor,
                                video_message(video, prompt, nframes=self.nframes),
                                self._device)
        if self.ctl is None:
            return generate(self._model, self.processor, inputs, max_new_tokens)
        with controls(self._model, **self.ctl):
            return generate(self._model, self.processor, inputs, max_new_tokens)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        for req in tqdm(requests, disable=self.rank != 0, desc="Model Responding"):
            context, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            visuals = doc_to_visual(self.task_dict[task][split][doc_id]) or []
            videos = [v for v in visuals
                      if isinstance(v, str) and v.lower().endswith(VIDEO_EXTS)]
            until = gen_kwargs.get("until", [])
            until = [until] if isinstance(until, str) else until
            mnt = self.force_max_new_tokens or int(gen_kwargs.get("max_new_tokens", 128))
            context = context.replace("<image>", "")
            ans = ""
            if not videos:
                eval_logger.error(f"[{task}] doc {doc_id}: no video ({visuals}); empty answer")
            else:
                try:
                    ans = self._answer(videos[0], context, mnt)
                except Exception as e:  # counted as wrong, run continues
                    eval_logger.error(f"[{task}] doc {doc_id}: {type(e).__name__}: {e}")
                    torch.cuda.empty_cache()
            for term in until:
                if term:
                    ans = ans.split(term)[0]
            res.append(ans)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
        return res
