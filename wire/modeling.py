"""WIRE on Qwen3-VL: compression, sequence contraction, generated updates.

`attach_wire(model, cfg)` patches a stock `Qwen3VLForConditionalGeneration`
in place:

  * a wrapper on `model.model.language_model.forward` contracts the decoder
    input before layer 0 (paper Sec. 3.3): every frame run of eta_t visual
    positions is replaced by the k_t tokens the compressor produced for it,
    and the other positions are deleted from the embeddings, the attention
    mask and the rotary index tensor;
  * every adapted decoder layer gets a forward that, during prefill, runs its
    hypernetwork and applies the generated rank-r update to q/k/v/o and
    gate/up/down functionally (Eq. 8). The updates are cached per layer and
    reused during decoding, or merged into the weights (`merge_updates`);
  * the trainable modules live in `model.wire` (an `nn.ModuleDict` with
    `hypernets`, `compressor` and `cross_pool`); everything else is frozen.

The per-forward bookkeeping lives in `model.wire_state` (a `WireState`).
Ablation switches are set with the `controls` context manager.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
from types import MethodType
from typing import Dict, List, Optional

import torch
from torch import nn

from .compressor import (SelectiveScanCompressor, bilinear_queries,
                         contraction_segments, frame_budgets,
                         video_frame_runs)
from .config import WireConfig
from .hypernet import FrameCrossPool, LayerHyperNet

ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")


class WireState:
    """Shared, mutable state read by the patched forwards."""

    def __init__(self, cfg: WireConfig):
        self.cfg = cfg
        self.k = cfg.tokens_kept_per_frame
        self.tokens_per_frame = cfg.tokens_per_frame
        self.scaling = cfg.scaling
        self.cond_on_queries = cfg.cond_on_queries
        self.video_token_id: Optional[int] = None
        self.image_token_id: Optional[int] = None
        self.compressor: Optional[SelectiveScanCompressor] = None
        self.cross_pool: Optional[FrameCrossPool] = None

        # --- controls (see `controls`) -------------------------------------
        self.zero_context = cfg.zero_context
        self.no_update = False
        self.frame_drop = 0.0
        self.frame_drop_seed = 0
        self.swap_condition = False
        self.scan_bidirectional = True
        self.scan_enabled = True            # False: bilinear queries only
        self.teacher_mode = False           # run the unmodified backbone

        # --- per-forward ---------------------------------------------------
        self.video_mask: Optional[torch.Tensor] = None
        self.prompt_end_pos: Optional[torch.Tensor] = None
        self.video_grid = None
        self.contract_idx = None            # (idx [B, S~], pad_mask [B, S~])
        self.queries: Optional[List[torch.Tensor]] = None
        self.kv_caches: Optional[list] = None
        self.updates: Dict[int, Optional[dict]] = {}
        self.merged = False
        self._merge_undo: list = []

        # --- diagnostics (weight-space analysis, update swaps) -------------
        self.cond_record: Optional[dict] = None   # layer -> hypernet input
        self.update_override: Optional[dict] = None  # layer -> update


@contextlib.contextmanager
def controls(model, **kw):
    """Temporarily set ablation switches, e.g.

        with controls(model, no_update=True): ...
        with controls(model, zero_context=True): ...
        with controls(model, frame_drop=0.5, frame_drop_seed=1): ...
        with controls(model, no_update=True, scan_enabled=False): ...
    """
    st = model.wire_state
    old = {k: getattr(st, k) for k in kw}
    for k, v in kw.items():
        if not hasattr(st, k):
            raise AttributeError(f"unknown control {k!r}")
        setattr(st, k, v)
    try:
        yield st
    finally:
        for k, v in old.items():
            setattr(st, k, v)


def set_prompt_end(model, input_ids: torch.Tensor, labels: torch.Tensor):
    """Training only: the hypernetwork reads the hidden state at the last
    prompt position p. Under teacher forcing the response is in the same
    sequence, so p is taken from the labels (last position with -100). At
    inference p is simply the last prefill position."""
    st = model.wire_state
    pos = torch.arange(labels.shape[1], device=labels.device)
    pe = torch.where(labels == -100, pos, torch.full_like(pos, -1)).max(1).values
    st.prompt_end_pos = torch.where(pe >= 0, pe, torch.zeros_like(pe))
    vm = (input_ids == st.video_token_id) | (input_ids == st.image_token_id)
    st.video_mask = vm if bool(vm.any()) else None


# ----------------------------------------------------------------------------
# sequence contraction (Sec. 3.3)
# ----------------------------------------------------------------------------
def _contracted_forward(st: WireState, orig_forward, input_ids=None,
                        position_ids=None, attention_mask=None,
                        past_key_values=None, inputs_embeds=None,
                        visual_pos_masks=None, deepstack_visual_embeds=None,
                        **kwargs):
    ds_kwargs = {}
    if visual_pos_masks is not None or deepstack_visual_embeds is not None:
        ds_kwargs = {"visual_pos_masks": visual_pos_masks,
                     "deepstack_visual_embeds": deepstack_visual_embeds}
    vm = st.video_mask
    if (st.teacher_mode or vm is None or inputs_embeds is None
            or vm.shape[1] != inputs_embeds.shape[1]):
        # teacher pass, text-only input, or a decoding step
        st.contract_idx = None
        return orig_forward(input_ids=input_ids, position_ids=position_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, **ds_kwargs, **kwargs)

    B, S, D = inputs_embeds.shape
    dev = inputs_embeds.device
    embeds = inputs_embeds.clone()
    keep = ~vm
    zero_ctx = bool(st.zero_context)
    collect_queries = zero_ctx or st.cond_on_queries
    queries, kv_caches = [], []
    pe_full = st.prompt_end_pos
    ds_pooled = ([[] for _ in deepstack_visual_embeds]
                 if deepstack_visual_embeds is not None else None)
    ds_off = 0
    tpf = max(1, int(st.tokens_per_frame))

    for b in range(B):
        vpos = vm[b].nonzero(as_tuple=True)[0]
        n = int(vpos.numel())
        ve = embeds[b, vpos]
        run_id, run_len = video_frame_runs(vpos, tpf)
        R = int(run_len.numel())
        run_start = torch.cumsum(run_len, 0) - run_len
        kt = frame_budgets(run_len, st.k, tpf)

        # segment map: frame run -> k_t consecutive near-equal segments
        seg, last = contraction_segments(run_id, run_len, kt)
        m = int(kt.sum())
        cnt = torch.zeros(m, device=dev, dtype=torch.float32)
        cnt.index_add_(0, seg, torch.ones_like(seg, dtype=torch.float32))

        # compressor: bilinear queries refined by the scan
        q0 = bilinear_queries(ve, run_start, run_len, kt, st.video_grid)
        q_run_id = torch.repeat_interleave(torch.arange(R, device=dev), kt)
        text = None
        if st.cfg.scan_with_text:
            tmask = ~vm[b]
            if pe_full is not None:                  # prompt, not response
                tmask &= torch.arange(S, device=dev) <= pe_full[b]
            ti = tmask.nonzero(as_tuple=True)[0]
            if ti.numel():
                text = embeds[b, ti]
        if st.scan_enabled:
            q, rows, rows_run = st.compressor(
                ve, q0.to(ve.dtype), run_id, q_run_id, text=text,
                bidirectional=st.scan_bidirectional)
            kv_caches.append(st.cross_pool.prepare(rows, rows_run, q_run_id))
            q = q.float()
        else:                               # training-free bilinear pooling
            q = q0
            kv_caches.append(None)
        if collect_queries:
            queries.append(q.to(embeds.dtype))

        # each segment's token goes to the segment's last position
        slots = vpos[last]

        # frame dropout: hide a random subset of frames from the decoder
        # input; the hypernetworks still see every frame
        seg_keep = None
        if st.frame_drop > 0.0 and not zero_ctx:
            g = torch.Generator()
            g.manual_seed(int(st.frame_drop_seed) * 8192 + b)
            drop = torch.rand(R, generator=g) < st.frame_drop
            if bool(drop.all()):
                drop[int(torch.randint(R, (1,), generator=g))] = False
            seg_keep = ~drop.to(dev)[q_run_id]

        if not zero_ctx:
            if seg_keep is None:
                embeds[b, slots] = q.to(embeds.dtype)
                keep[b, slots] = True
            else:
                embeds[b, slots[seg_keep]] = q.to(embeds.dtype)[seg_keep]
                keep[b, slots[seg_keep]] = True
            if ds_pooled is not None:
                # deepstack features are averaged over the same segments
                for i, ds in enumerate(deepstack_visual_embeds):
                    s = torch.zeros(m, D, device=dev, dtype=torch.float32)
                    s.index_add_(0, seg, ds[ds_off:ds_off + n].float())
                    row = (s / cnt.clamp(min=1).unsqueeze(-1)).to(ds.dtype)
                    ds_pooled[i].append(row if seg_keep is None
                                        else row[seg_keep])
                ds_off += n

    if collect_queries and st.swap_condition and B > 1:
        # condition every sample on the next sample's video
        queries = queries[1:] + queries[:1]
        kv_caches = kv_caches[1:] + kv_caches[:1]
    st.queries = queries if collect_queries else None
    st.kv_caches = kv_caches

    # gather the kept positions (right padded with S - 1, masked out)
    lens = keep.sum(dim=1)
    S_c = int(lens.max())
    idx = torch.stack([
        torch.cat([keep[b].nonzero(as_tuple=True)[0],
                   torch.full((S_c - int(lens[b]),), S - 1, device=dev,
                              dtype=torch.long)])
        for b in range(B)])
    pad_mask = torch.arange(S_c, device=dev)[None, :] < lens[:, None]
    embeds_c = torch.gather(embeds, 1, idx.unsqueeze(-1).expand(-1, -1, D))
    am = attention_mask if attention_mask is not None else \
        torch.ones(B, S, dtype=torch.long, device=dev)
    am_c = torch.gather(am, 1, idx) * pad_mask.to(am.dtype)
    pos_c = None
    if position_ids is not None:
        # positions are gathered, not renumbered (Eq. 10)
        g = idx.unsqueeze(0).expand(position_ids.shape[0], -1, -1) \
            if position_ids.dim() == 3 else idx
        pos_c = torch.gather(position_ids, -1, g)

    st.contract_idx = (idx, pad_mask)
    st.video_mask = torch.gather(vm, 1, idx) & pad_mask
    if st.prompt_end_pos is not None:
        pos_full = torch.arange(S, device=dev)[None, :]
        pe_c = (keep & (pos_full <= st.prompt_end_pos[:, None])).sum(1) - 1
        st.prompt_end_pos = pe_c.clamp(min=0)

    if ds_kwargs:
        ds_kwargs = {
            "visual_pos_masks": (st.video_mask if visual_pos_masks is not None
                                 and not zero_ctx else None),
            "deepstack_visual_embeds": (
                [torch.cat(ch, dim=0) for ch in ds_pooled]
                if ds_pooled is not None and not zero_ctx else None),
        }
    return orig_forward(input_ids=None, position_ids=pos_c,
                        attention_mask=am_c, past_key_values=past_key_values,
                        inputs_embeds=embeds_c, **ds_kwargs, **kwargs)


# ----------------------------------------------------------------------------
# generated updates (Sec. 3.4, 3.5)
# ----------------------------------------------------------------------------
def _lora_linear(linear: nn.Linear, x, A, B, scaling):
    """y = x W^T + scaling * (x A^T) B, with per-sample A [B, r, d_in] and
    B [B, r, d_out] (Eq. 8)."""
    y = linear(x)
    xa = torch.einsum("bsd,brd->bsr", x.to(A.dtype), A)
    return y + (scaling * torch.einsum("bsr,bro->bso", xa, B)).to(y.dtype)


def _attention_with_update(attn, x, position_embeddings, attention_mask,
                           past_key_values, upd, scaling):
    """Qwen3-VL attention forward with the generated update applied."""
    mod = importlib.import_module(type(attn).__module__)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    input_shape = x.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    q = _lora_linear(attn.q_proj, x, *upd["q_proj"], scaling)
    k = _lora_linear(attn.k_proj, x, *upd["k_proj"], scaling)
    v = _lora_linear(attn.v_proj, x, *upd["v_proj"], scaling)
    q = attn.q_norm(q.view(hidden_shape)).transpose(1, 2)
    k = attn.k_norm(k.view(hidden_shape)).transpose(1, 2)
    v = v.view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = mod.apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, attn.layer_idx)
    attn_fn = ALL_ATTENTION_FUNCTIONS.get_interface(
        attn.config._attn_implementation, mod.eager_attention_forward)
    out, _ = attn_fn(attn, q, k, v, attention_mask, dropout=0.0,
                     scaling=attn.scaling)
    out = out.reshape(*input_shape, -1).contiguous()
    return _lora_linear(attn.o_proj, out, *upd["o_proj"], scaling)


def _mlp_with_update(mlp, x, upd, scaling):
    g = _lora_linear(mlp.gate_proj, x, *upd["gate_proj"], scaling)
    u = _lora_linear(mlp.up_proj, x, *upd["up_proj"], scaling)
    return _lora_linear(mlp.down_proj, mlp.act_fn(g) * u,
                        *upd["down_proj"], scaling)


def _hypernet_input(st: WireState, hidden_states: torch.Tensor):
    """h = concat(h_v, h_q, h_c) for one layer (Sec. 3.4)."""
    B, S, _ = hidden_states.shape
    vm = st.video_mask
    vm_f = vm.unsqueeze(-1).to(hidden_states.dtype)
    h_v = (hidden_states * vm_f).sum(1) / vm_f.sum(1).clamp(min=1)
    q = st.queries
    if q is not None and len(q) == B:
        # input dropout model / zero context: mean of the refined queries
        h_v = torch.stack([x.float().mean(0) for x in q], 0) \
            .to(hidden_states.dtype)
    pe = st.prompt_end_pos
    if pe is None:
        pe = torch.full((B,), S - 1, device=hidden_states.device,
                        dtype=torch.long)
    h_q = hidden_states[torch.arange(B), pe.clamp(0, S - 1)]
    h_c = []
    for b in range(B):
        cache = st.kv_caches[b] if st.kv_caches is not None else None
        if q is not None and len(q) == B:
            q_tok = q[b].to(hidden_states.dtype)
        else:
            q_tok = hidden_states[b, vm[b].nonzero(as_tuple=True)[0]]
        if cache is not None and q_tok.shape[0] == cache[2].numel():
            h_c.append(st.cross_pool(q_tok, cache))
        else:
            h_c.append(h_v[b])
    h = torch.cat([h_v, h_q], dim=-1)
    return torch.cat([h, torch.stack(h_c, 0).to(h.dtype)], dim=-1)


def _decoder_layer_forward(self, hidden_states, position_embeddings=None,
                           attention_mask=None, position_ids=None,
                           past_key_values=None, **kwargs):
    """Replacement forward for an adapted Qwen3-VL decoder layer."""
    st: WireState = self._wire_state
    li = self.self_attn.layer_idx
    residual = hidden_states
    S = hidden_states.shape[1]
    prefill = st.video_mask is not None and st.video_mask.shape[1] == S

    if st.teacher_mode:
        upd = None
    elif prefill and not st.no_update:
        h = _hypernet_input(st, hidden_states)
        if st.cond_record is not None:
            st.cond_record[li] = h.detach()
        upd = (st.update_override[li] if st.update_override is not None
               else self._wire_hypernet(h))
        st.updates[li] = upd
    elif prefill:
        upd = st.updates[li] = None
    else:
        upd = None if st.merged else st.updates.get(li)

    normed = self.input_layernorm(hidden_states)
    if upd is not None:
        attn_out = _attention_with_update(
            self.self_attn, normed, position_embeddings, attention_mask,
            past_key_values, upd, st.scaling)
    else:
        attn_out, _ = self.self_attn(
            hidden_states=normed, attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings, **kwargs)
    hidden_states = residual + attn_out
    residual = hidden_states
    normed = self.post_attention_layernorm(hidden_states)
    if upd is not None and st.cfg.adapt_mlp:
        return residual + _mlp_with_update(self.mlp, normed, upd, st.scaling)
    return residual + self.mlp(normed)


@torch.no_grad()
def merge_updates(model):
    """W <- W + Delta W for every adapted map, using the updates cached by
    the last prefill (batch size 1). Decoding then runs the plain backbone.
    Call `unmerge_updates` before the next video."""
    st = model.wire_state
    if st.merged:
        return
    layers = model.model.language_model.layers
    undo = []
    for li, upd in st.updates.items():
        if upd is None:
            continue
        layer = layers[li]
        for name, (A, Bf) in upd.items():
            assert A.shape[0] == 1, "merging needs batch size 1"
            target = getattr(layer.self_attn, name, None) or \
                getattr(layer.mlp, name)
            dw = (st.scaling * torch.einsum("ri,ro->oi", A[0].float(),
                                            Bf[0].float())
                  ).to(target.weight.dtype)
            target.weight.data += dw
            undo.append((target, dw))
    st.merged = True
    st._merge_undo = undo


@torch.no_grad()
def unmerge_updates(model):
    st = model.wire_state
    for target, dw in st._merge_undo:
        target.weight.data -= dw
    st._merge_undo = []
    st.merged = False


# ----------------------------------------------------------------------------
# attach
# ----------------------------------------------------------------------------
def attach_wire(model, cfg: WireConfig):
    """Patch a `Qwen3VLForConditionalGeneration` in place and return it.

    Creates `model.wire` (trainable modules) and `model.wire_state`, freezes
    the backbone, and keeps the hypernetworks in float32 (training masters)
    and the compressor in bfloat16. Use `model.wire.to(dtype)` for inference.
    """
    lm = model.model.language_model
    tcfg = lm.config
    hidden = tcfg.hidden_size
    st = WireState(cfg)
    st.video_token_id = model.config.video_token_id
    st.image_token_id = model.config.image_token_id
    dev = next(lm.parameters()).device

    hypernets = nn.ModuleDict()
    for i in cfg.target_layers(tcfg.num_hidden_layers):
        layer = lm.layers[i]
        names = ATTN_MODULES + (MLP_MODULES if cfg.adapt_mlp else ())
        dims = {}
        for nm in names:
            lin = getattr(layer.self_attn, nm, None) or getattr(layer.mlp, nm)
            dims[nm] = (lin.in_features, lin.out_features)
        hn = LayerHyperNet(3 * hidden, dims, rank=cfg.rank,
                           trunk_width=cfg.trunk_width,
                           latent_size=cfg.latent_size).to(dev)
        hypernets[str(i)] = hn
        layer._wire_state = st
        layer._wire_hypernet = hn
        layer.forward = MethodType(_decoder_layer_forward, layer)

    compressor = SelectiveScanCompressor(
        hidden, d_state=cfg.scan_d_state, d_conv=cfg.scan_d_conv,
        expand=cfg.scan_expand, headdim=cfg.scan_headdim,
        n_layer=cfg.scan_layers,
        out_proj_init_std=cfg.out_proj_init_std).to(dev)
    cross_pool = FrameCrossPool(hidden, dim=cfg.cross_pool_dim,
                                heads=cfg.cross_pool_heads,
                                out_proj_init_std=cfg.out_proj_init_std).to(dev)
    st.compressor, st.cross_pool = compressor, cross_pool

    model.wire = nn.ModuleDict({"hypernets": hypernets,
                                "compressor": compressor,
                                "cross_pool": cross_pool})
    model.wire_state = st

    for p in model.parameters():
        p.requires_grad = False
    for p in model.wire.parameters():
        p.requires_grad = True
        p.data = p.data.float()
    for p in compressor.parameters():      # the scan trains in bfloat16
        p.data = p.data.to(torch.bfloat16)

    lm_forward = lm.forward

    @functools.wraps(lm_forward)
    def contracted(*args, **kw):
        return _contracted_forward(st, lm_forward, *args, **kw)

    lm.forward = contracted

    model_forward = model.forward

    @functools.wraps(model_forward)
    def forward(self, *args, **kwargs):
        if st.teacher_mode:
            st.video_mask = None
            return model_forward(*args, **kwargs)
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        vm = None
        if input_ids is not None and input_ids.dim() == 2 \
                and input_ids.shape[1] > 1:
            vm = (input_ids == st.video_token_id) | \
                (input_ids == st.image_token_id)
            if not bool(vm.any()):
                vm = None
        # kept after the call: activation checkpointing recomputes the
        # layers during backward and must see the same mask
        st.video_mask = vm
        grid = kwargs.get("video_grid_thw")
        if grid is not None and len(grid):
            merge = int(getattr(self.config.vision_config,
                                "spatial_merge_size", 2))
            st.video_grid = (int(grid[0][1]) // merge,
                             int(grid[0][2]) // merge)
        return model_forward(*args, **kwargs)

    model.forward = MethodType(forward, model)
    return model
