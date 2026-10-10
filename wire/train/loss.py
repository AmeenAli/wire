"""Self-distillation objective (paper Sec. 3.6).

Teacher: the frozen backbone with the full visual sequence (56 tokens per
frame, no generated update). Student: the same backbone with the contracted
sequence and the generated update. On the answer tokens

    L = tau^2 KL(p_teacher || p_student) + lambda_ce CE(student, answer)

normalized by the number of answer tokens in the global batch. The input-
dropout recipe adds three regularizers, drawn per micro-batch from an RNG
shared by all ranks (every rank must take the same branch, or DDP hangs):

  * context dropout: with probability p the student sees no visual token
    (the video reaches the decoder only through the update);
  * frame dropout: on the other micro-batches, with probability p, a
    U(lo, hi) fraction of the frames is hidden from the student's decoder
    input (the hypernetworks still see every frame);
  * swap margin: with probability p, a third pass conditions the update on
    the next sample's video and a hinge asks the mismatched KL to exceed the
    own KL by `margin` nats per token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..modeling import set_prompt_end


@dataclass
class DistillConfig:
    tau: float = 1.0
    ce_weight: float = 0.1
    chunk: int = 1024
    # input-dropout recipe (all 0 = plain distillation)
    context_dropout: float = 0.0
    frame_drop_p: float = 0.0
    frame_drop_lo: float = 0.3
    frame_drop_hi: float = 0.7
    swap_coef: float = 0.0
    swap_margin: float = 0.2
    swap_p: float = 0.5
    swap_max_tokens: int = 6000
    seed: int = 20260901
    # the first 1,600 steps of the paper's input-dropout run seeded the
    # dropout RNG per rank (seed + rank); everything later shares one stream
    per_rank_seed: bool = False


class VisionFeatureCache:
    """The teacher and the student see the same pixels and the vision tower
    is frozen: encode once per step, reuse the (detached) features."""

    def __init__(self, inner):
        self.orig = inner.get_video_features
        self.key = self.val = None
        inner.get_video_features = self

    def clear(self):
        self.key = self.val = None

    def __call__(self, *args, **kwargs):
        pv = args[0] if args else kwargs.get("pixel_values_videos")
        key = (pv.data_ptr(), tuple(pv.shape)) if pv is not None else None
        if key is not None and key == self.key:
            return self.val
        out = _detach(self.orig(*args, **kwargs))
        if key is not None:
            self.key, self.val = key, out
        return out


def _detach(o):
    if torch.is_tensor(o):
        return o.detach()
    if isinstance(o, (list, tuple)):
        return type(o)(_detach(x) for x in o)
    from transformers.utils import ModelOutput
    if isinstance(o, ModelOutput):
        return o.__class__(**{k: _detach(v) for k, v in o.items()})
    return o


def _kl_ce(lm_head, hs, ht, gt, cfg: DistillConfig, with_ce: bool):
    """Chunked KL(teacher || student) (+ CE) summed over tokens."""
    kl = hs.new_zeros((), dtype=torch.float32)
    ce = hs.new_zeros((), dtype=torch.float32)
    for i in range(0, hs.shape[0], cfg.chunk):
        ls = lm_head(hs[i:i + cfg.chunk]).float()
        with torch.no_grad():
            lt = lm_head(ht[i:i + cfg.chunk]).float()
        kl = kl + F.kl_div(F.log_softmax(ls / cfg.tau, -1),
                           F.log_softmax(lt / cfg.tau, -1),
                           log_target=True, reduction="sum") * cfg.tau ** 2
        if with_ce:
            ce = ce + F.cross_entropy(ls, gt[i:i + cfg.chunk], reduction="sum")
    return kl, ce


class DistillLoss:
    """`compute(trainer, model, inputs, num_items_in_batch)` -> loss."""

    _STATE_KEYS = ("video_mask", "prompt_end_pos", "contract_idx", "queries",
                   "kv_caches", "video_grid")

    def __init__(self, cfg: DistillConfig, rank: int = 0):
        self.cfg = cfg
        self.gen = torch.Generator().manual_seed(
            cfg.seed + (rank if cfg.per_rank_seed else 0))
        self.cache = None
        self.stats = {"n": 0, "kl": 0.0, "ce": 0.0, "n_zc": 0, "n_fd": 0,
                      "n_swap": 0, "gap": 0.0}

    def _rand(self) -> float:
        return float(torch.rand((), generator=self.gen))

    def compute(self, trainer, model, inputs, num_items_in_batch=None):
        cfg = self.cfg
        labels = inputs.pop("labels")
        net = trainer.accelerator.unwrap_model(model)
        st = net.wire_state
        if self.cache is None:
            self.cache = VisionFeatureCache(net.model)
        self.cache.clear()

        # regimes for this micro-batch; set now and left in place so that
        # activation recomputation during backward sees the same flags
        # (the k = 0 model keeps zero_context on through its config)
        if cfg.context_dropout > 0:
            st.zero_context = self._rand() < cfg.context_dropout
        st.frame_drop = 0.0
        if cfg.frame_drop_p > 0 and not st.zero_context \
                and self._rand() < cfg.frame_drop_p:
            st.frame_drop = cfg.frame_drop_lo + \
                (cfg.frame_drop_hi - cfg.frame_drop_lo) * self._rand()
            st.frame_drop_seed = int(torch.randint(2 ** 31 - 1, (1,),
                                                   generator=self.gen))

        lm_head = net.lm_head
        net.lm_head = nn.Identity()          # forwards return hidden states
        h_s2 = None
        try:
            # teacher first (full sequence, no update, no grad) ...
            st.teacher_mode = True
            st.updates = {}
            with torch.no_grad():
                h_t = model(**inputs).logits
            st.teacher_mode = False
            # ... student last, so its masks stay installed for backward
            set_prompt_end(net, inputs["input_ids"], labels)
            h_s = model(**inputs).logits
            snap = {k: getattr(st, k) for k in self._STATE_KEYS}

            # the draw happens before the skip conditions so that the shared
            # stream stays aligned across ranks
            swap_draw = self._rand() < cfg.swap_p
            do_swap = (cfg.swap_coef > 0 and swap_draw and labels.shape[0] > 1
                       and h_s.shape[1] <= cfg.swap_max_tokens)
            if cfg.swap_coef > 0 and swap_draw and torch.distributed.is_initialized():
                flag = h_s.new_tensor([float(do_swap)])
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
                do_swap = bool(flag.item() > 0)
            if do_swap:
                # run without activation checkpointing: a checkpointed pass
                # would re-read swap_condition after it has been reset
                was_gc = bool(getattr(net, "is_gradient_checkpointing", False))
                if was_gc:
                    net.gradient_checkpointing_disable()
                st.swap_condition = True
                try:
                    set_prompt_end(net, inputs["input_ids"], labels)
                    h_s2 = model(**inputs).logits
                finally:
                    st.swap_condition = False
                    if was_gc:
                        net.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False})
            for k, v in snap.items():          # the own pass's backward
                setattr(st, k, v)
        finally:
            st.teacher_mode = False
            st.swap_condition = False
            net.lm_head = lm_head

        # answer tokens: logits at t predict labels at t + 1
        lab = labels[:, 1:]
        sel = lab != -100
        idx, pad_mask = st.contract_idx
        lab_s = torch.where(pad_mask, torch.gather(labels, 1, idx),
                            torch.full_like(idx, -100))[:, 1:]
        sel_s = lab_s != -100
        hs = h_s[:, :-1][sel_s].to(lm_head.weight.dtype)
        ht = h_t[:, :-1][sel].to(lm_head.weight.dtype)
        gt = lab[sel]
        assert hs.shape[0] == ht.shape[0], (hs.shape, ht.shape)
        n_tok = hs.shape[0]
        if n_tok == 0:
            return h_s.sum() * 0.0

        kl, ce = _kl_ce(lm_head, hs, ht, gt, cfg, cfg.ce_weight > 0)
        denom = float(num_items_in_batch) if num_items_in_batch is not None \
            else float(n_tok)
        loss = (kl + cfg.ce_weight * ce) / denom
        if h_s2 is not None:
            hs2 = h_s2[:, :-1][sel_s].to(lm_head.weight.dtype)
            kl2, _ = _kl_ce(lm_head, hs2, ht, gt, cfg, False)
            gap = (kl2 - kl) / n_tok
            loss = loss + cfg.swap_coef * F.relu(cfg.swap_margin - gap) * (n_tok / denom)
            self.stats["n_swap"] += 1
            self.stats["gap"] += float(gap)

        s = self.stats
        s["n"] += 1
        s["kl"] += float(kl) / n_tok
        s["ce"] += float(ce) / n_tok
        s["n_zc"] += int(bool(st.zero_context))
        s["n_fd"] += int(st.frame_drop > 0)
        return loss

    def pop_stats(self) -> dict:
        s, n = self.stats, max(self.stats["n"], 1)
        out = {"kl_per_token": s["kl"] / n, "ce_per_token": s["ce"] / n,
               "zero_context_frac": s["n_zc"] / n,
               "frame_drop_frac": s["n_fd"] / n,
               "swap_frac": s["n_swap"] / n}
        if s["n_swap"]:
            out["swap_gap"] = s["gap"] / s["n_swap"]
        self.stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in s.items()}
        return out
