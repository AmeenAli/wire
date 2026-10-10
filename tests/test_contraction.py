"""Frame budgets, frame runs and the segment map of the contraction (Sec. 3.3)."""

import random

import pytest
import torch

from wire.compressor import (_aspect_factor, _interleave_mask, bilinear_queries,
                             contraction_segments, frame_budgets, video_frame_runs)


def reference_segments(run_id, run_len, kt):
    """The segment map as it was written inline in wire/modeling.py."""
    n, dev = run_id.numel(), run_id.device
    run_start = torch.cumsum(run_len, 0) - run_len
    koff = torch.cumsum(kt, 0) - kt
    jin = torch.arange(n, device=dev) - run_start[run_id]
    seg = koff[run_id] + (jin * kt[run_id]) // run_len[run_id]
    m = int(kt.sum())
    last = torch.searchsorted(seg, torch.arange(1, m + 1, device=dev)) - 1
    return seg, last


def layout(widths, gap=3, lead=5):
    """Visual positions of frame runs of the given widths, `gap` text
    positions apart."""
    pos, cur = [], lead
    for w in widths:
        pos.extend(range(cur, cur + w))
        cur += w + gap
    return torch.tensor(pos)


def test_frame_budgets_follow_eq3():
    run_len = torch.tensor([56, 112, 39, 1, 3])
    assert frame_budgets(run_len, 16, 56).tolist() == [16, 32, 11, 1, 1]
    for k in (16, 12, 8, 6):
        assert frame_budgets(torch.tensor([56]), k, 56).tolist() == [k]
    # at least one token per run and never more than the run has
    assert frame_budgets(torch.tensor([3]), 100, 56).tolist() == [3]


def test_frame_runs_split_at_text():
    run_id, run_len = video_frame_runs(layout([56, 56, 50]), 56)
    assert run_len.tolist() == [56, 56, 50]
    assert torch.equal(run_id, torch.repeat_interleave(torch.arange(3), run_len))


def test_frame_runs_fall_back_to_fixed_chunks():
    _, run_len = video_frame_runs(torch.arange(10, 122), 56)
    assert run_len.tolist() == [56, 56]
    _, run_len = video_frame_runs(torch.arange(40), 56)
    assert run_len.tolist() == [40]


@pytest.mark.parametrize("seed", range(20))
def test_segments_match_the_model(seed):
    rng = random.Random(seed)
    widths = [rng.randint(20, 130) for _ in range(rng.randint(1, 12))]
    k = rng.choice([6, 8, 12, 16])
    run_id, run_len = video_frame_runs(layout(widths), 56)
    kt = frame_budgets(run_len, k, 56)
    seg, last = contraction_segments(run_id, run_len, kt)
    ref_seg, ref_last = reference_segments(run_id, run_len, kt)
    assert torch.equal(seg, ref_seg) and torch.equal(last, ref_last)


@pytest.mark.parametrize("k,ends", [
    (16, [3, 6, 10, 13, 17, 20, 24, 27, 31, 34, 38, 41, 45, 48, 52, 55]),
    (12, [4, 9, 13, 18, 23, 27, 32, 37, 41, 46, 51, 55]),
    (8, [6, 13, 20, 27, 34, 41, 48, 55]),
    (6, [9, 18, 27, 37, 46, 55]),
])
def test_last_position_of_each_segment(k, ends):
    run_id, run_len = video_frame_runs(torch.arange(56), 56)
    _, last = contraction_segments(run_id, run_len, frame_budgets(run_len, k, 56))
    assert last.tolist() == ends


def test_segments_cover_each_run():
    widths = [56, 49, 63, 112]
    run_id, run_len = video_frame_runs(layout(widths), 56)
    kt = frame_budgets(run_len, 16, 56)
    seg, last = contraction_segments(run_id, run_len, kt)
    counts = torch.bincount(seg, minlength=int(kt.sum()))
    assert int(counts.min()) >= 1                       # no empty segment
    assert torch.equal(seg[last], torch.arange(int(kt.sum())))
    ends = torch.cumsum(run_len, 0) - 1                 # each run ends on a kept position
    assert set(ends.tolist()) <= set(last.tolist())


def test_paper_contraction_at_256_frames():
    from cpu_check import contraction_lengths
    assert contraction_lengths(16)["contracted"] == 5154
    assert contraction_lengths(6)["contracted"] == 2594


def test_aspect_factor():
    assert _aspect_factor(16, 7, 8) == (4, 4)
    assert _aspect_factor(12, 7, 8) == (3, 4)
    assert _aspect_factor(32, 8, 14) == (4, 8)


@pytest.mark.parametrize("n,m", [(56, 16), (112, 32), (39, 11), (10, 0), (1, 5)])
def test_interleave_mask_counts(n, m):
    mask = _interleave_mask(n, m, "cpu")
    assert mask.numel() == n + m and int(mask.sum()) == m


def test_bilinear_queries_shapes_and_constant_frames():
    D = 8
    # equal frames on a 7 x 8 grid: the batched path
    run_len = torch.tensor([56, 56, 56])
    kt = frame_budgets(run_len, 16, 56)
    run_start = torch.cumsum(run_len, 0) - run_len
    ve = torch.repeat_interleave(torch.arange(3.0), 56).unsqueeze(-1).repeat(1, D)
    q = bilinear_queries(ve, run_start, run_len, kt, (7, 8))
    assert q.shape == (48, D)
    # a constant frame resamples to the same constant
    assert torch.allclose(q, torch.repeat_interleave(torch.arange(3.0), 16).unsqueeze(-1).repeat(1, D))
    # frames of different widths: one frame at a time
    run_len = torch.tensor([56, 49])
    kt = frame_budgets(run_len, 16, 56)
    run_start = torch.cumsum(run_len, 0) - run_len
    q = bilinear_queries(torch.randn(105, D), run_start, run_len, kt, (7, 8))
    assert q.shape == (int(kt.sum()), D)
