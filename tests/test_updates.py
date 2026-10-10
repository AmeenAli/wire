"""Hypernetworks, the cross attention readout and the merge (Sec. 3.4, 3.5)."""

import torch
from torch import nn

from wire.config import WireConfig
from wire.hypernet import FrameCrossPool, LayerHyperNet
from wire.modeling import (WireState, _lora_linear, controls, merge_updates,
                           unmerge_updates)

DIMS = {"q_proj": (32, 32), "k_proj": (32, 16), "down_proj": (48, 32)}


def test_hypernet_shapes_and_zero_update_at_init():
    hn = LayerHyperNet(96, DIMS, rank=4, trunk_width=16, latent_size=8)
    out = hn(torch.randn(2, 96))
    for name, (d_in, d_out) in DIMS.items():
        A, B = out[name]
        assert A.shape == (2, 4, d_in) and B.shape == (2, 4, d_out)
        assert torch.count_nonzero(B) == 0             # V_m starts at zero


def test_cross_pool_reads_its_own_frames():
    pool = FrameCrossPool(16, dim=8, heads=2)
    rows, rows_run = torch.randn(6, 16), torch.tensor([0, 0, 0, 1, 1, 1])
    q_tok, q_run = torch.randn(4, 16), torch.tensor([0, 0, 1, 1])
    cache = pool.prepare(rows, rows_run, q_run)
    assert cache is not None
    out = pool(q_tok, cache)
    assert out.shape == (16,)
    # the output projection starts at zero: the readout is the plain mean
    assert torch.allclose(out, q_tok.mean(0), atol=1e-6)
    # frames of different widths fall back to the mean
    assert pool.prepare(rows, torch.tensor([0, 0, 0, 0, 1, 1]), q_run) is None
    assert torch.equal(pool(q_tok, None), q_tok.mean(0))


def test_merge_matches_adapted_maps_and_unmerge_restores():
    from cpu_check import merge_check
    r = merge_check(seed=1)
    assert r["step_zero"] == 0.0
    assert r["rank"] <= r["rank_limit"]
    assert r["merged_gap"] < 1e-4 and r["restored_gap"] < 1e-5


def test_lora_linear_is_the_low_rank_sum():
    torch.manual_seed(0)
    lin = nn.Linear(12, 10, bias=False)
    A, B = torch.randn(1, 3, 12), torch.randn(1, 3, 10)
    x = torch.randn(1, 5, 12)
    dw = 2.0 * B[0].T @ A[0]
    expected = x @ (lin.weight + dw).T
    assert torch.allclose(_lora_linear(lin, x, A, B, 2.0), expected, atol=1e-5)


def test_merge_twice_is_a_no_op():
    from cpu_check import _Layer, layer_dims
    dims = layer_dims(dict(hidden=16, heads=2, kv_heads=1, head_dim=8, intermediate=24))
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.layers = nn.ModuleList([_Layer(dims)])
    model.wire_state = st = WireState(WireConfig(rank=2, alpha=4))
    st.updates[0] = {n: (torch.randn(1, 2, i), torch.randn(1, 2, o)) for n, (i, o) in dims.items()}
    w0 = model.model.language_model.layers[0].self_attn.q_proj.weight.detach().clone()
    merge_updates(model)
    w1 = model.model.language_model.layers[0].self_attn.q_proj.weight.detach().clone()
    merge_updates(model)
    assert torch.equal(model.model.language_model.layers[0].self_attn.q_proj.weight, w1)
    unmerge_updates(model)
    assert torch.allclose(model.model.language_model.layers[0].self_attn.q_proj.weight, w0, atol=1e-6)


def test_controls_restore_the_switches():
    model = nn.Module()
    model.wire_state = st = WireState(WireConfig())
    with controls(model, no_update=True, frame_drop=0.5):
        assert st.no_update and st.frame_drop == 0.5
    assert not st.no_update and st.frame_drop == 0.0
    try:
        with controls(model, not_a_switch=True):
            pass
    except AttributeError:
        pass
    else:
        raise AssertionError("unknown switches must raise")
