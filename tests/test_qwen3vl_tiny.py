"""WIRE attached to a tiny, randomly initialized Qwen3-VL on CPU.

The selective scan needs the CUDA kernels of mamba-ssm, so a small residual
map stands in for it; everything else is the package's code path:
contraction, hypernetworks, cross attention readout, merge and decoding.
"""

import types

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration  # noqa: E402

import wire.modeling as modeling  # noqa: E402
from wire import WireConfig, attach_wire, controls  # noqa: E402
from wire.inference import generate  # noqa: E402

IMG, VID, VS, VE = 10, 11, 12, 13


class StubScan(nn.Module):
    """Stand-in for the Mamba2 compressor, with its interface."""

    def __init__(self, hidden, **kwargs):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(hidden)])
        self.mix = nn.Linear(hidden, hidden)

    def forward(self, video, query, run_id, q_run_id, text=None, bidirectional=True):
        q = query.to(self.mix.weight.dtype)
        return (q + self.mix(self.norms[0](q))).to(query.dtype), video, run_id


def tiny_inputs(groups=2, gh=4, gw=8, prompt=5, question=6):
    """Prompt, then per frame pair a timestamp and its visual tokens, then the
    question, laid out as the Qwen3-VL processor does."""
    g = torch.Generator().manual_seed(0)
    per = (gh // 2) * (gw // 2)
    ids, types_ = torch.randint(20, 60, (prompt,), generator=g).tolist(), [0] * prompt
    for _ in range(groups):
        ids += torch.randint(20, 60, (3,), generator=g).tolist() + [VS] + [VID] * per + [VE]
        types_ += [0] * 4 + [2] * per + [0]
    ids += torch.randint(20, 60, (question,), generator=g).tolist()
    types_ += [0] * question
    ids = torch.tensor([ids])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                pixel_values_videos=torch.randn(groups * gh * gw, 3 * 2 * 16 * 16, generator=g),
                video_grid_thw=torch.tensor([[groups, gh, gw]]),
                mm_token_type_ids=torch.tensor([types_])), per


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=8,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                          "mrope_section": [4, 2, 2], "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                           out_hidden_size=64, num_position_embeddings=64,
                           deepstack_visual_indexes=[0, 1]),
        image_token_id=IMG, video_token_id=VID, vision_start_token_id=VS, vision_end_token_id=VE)
    m = Qwen3VLForConditionalGeneration(cfg).eval()
    real = modeling.SelectiveScanCompressor
    modeling.SelectiveScanCompressor = StubScan
    try:
        attach_wire(m, WireConfig(tokens_kept_per_frame=4, tokens_per_frame=8, rank=4, alpha=8,
                                  trunk_width=32, latent_size=16, cross_pool_dim=32,
                                  cross_pool_heads=2))
    finally:
        modeling.SelectiveScanCompressor = real
    m.wire.to(torch.float32)
    for hn in m.wire["hypernets"].values():           # stand-in for trained weights
        for proj in hn.proj_B.values():
            nn.init.normal_(proj.weight, std=0.05)
    return m


def test_adapted_layers(model):
    assert sorted(int(k) for k in model.wire["hypernets"]) == [3, 7]


def test_contraction_keeps_text_and_segment_ends(model):
    x, per = tiny_inputs()
    S = x["input_ids"].shape[1]
    with torch.no_grad():
        model(**x, use_cache=True, logits_to_keep=1)
    idx, pad = model.wire_state.contract_idx
    kept = idx[0][pad[0]].tolist()
    video = (x["input_ids"][0] == VID).nonzero().flatten().tolist()
    text = [p for p in range(S) if p not in video]
    runs = [video[i:i + per] for i in range(0, len(video), per)]
    ends = [run[j] for run in runs for j in (1, 3, 5, 7)]   # 4 segments of 2 in a run of 8
    assert kept == sorted(text + ends)
    assert set(model.wire_state.updates) == {3, 7}


def test_merged_decoding_matches_the_adapted_maps(model):
    x, _ = tiny_inputs()
    tok = types.SimpleNamespace(eos_token_id=-1, decode=lambda ids, **kw: " ".join(map(str, ids)))
    proc = types.SimpleNamespace(tokenizer=tok)
    weights = {n: p.detach().clone() for n, p in model.model.language_model.named_parameters()}
    merged = generate(model, proc, x, max_new_tokens=6, merge=True)
    adapted = generate(model, proc, x, max_new_tokens=6, merge=False)
    assert merged == adapted and len(merged.split()) == 6
    for n, p in model.model.language_model.named_parameters():   # unmerged afterwards
        assert torch.allclose(p, weights[n], atol=1e-6), n


def test_no_update_switch(model):
    x, _ = tiny_inputs()
    with torch.no_grad():
        with controls(model, no_update=True):
            model(**x, use_cache=True, logits_to_keep=1)
            assert all(u is None for u in model.wire_state.updates.values())
        model(**x, use_cache=True, logits_to_keep=1)
    assert all(u is not None for u in model.wire_state.updates.values())
