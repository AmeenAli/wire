"""Configuration, released models, training data and the distillation loss."""

import json
import os

import torch
from torch import nn

from wire.config import CONFIG_NAME, WireConfig
from wire.models import MODELS, resolve


def test_adapted_layers_and_scaling():
    cfg = WireConfig()
    assert cfg.target_layers(36) == [3, 7, 11, 15, 19, 23, 27, 31, 35]
    assert WireConfig(layers=[5, 1, 5]).target_layers(36) == [1, 5]
    assert cfg.scaling == 2.0


def test_config_round_trip(tmp_path):
    cfg = WireConfig(tokens_kept_per_frame=6, zero_context=True)
    cfg.save(str(tmp_path))
    assert json.load(open(tmp_path / CONFIG_NAME))["tokens_kept_per_frame"] == 6
    assert WireConfig.load(str(tmp_path)) == cfg
    assert WireConfig.from_dict({**cfg.to_dict(), "unknown": 1}) == cfg


def test_released_models():
    assert {n: m["config"].tokens_kept_per_frame for n, m in MODELS.items()} == {
        "wire-k16": 16, "wire-k12": 12, "wire-k8": 8, "wire-k6": 6}
    assert resolve("checkpoints/wire-k16") == ("checkpoints/wire-k16", None)


def test_collate_pads_right():
    from wire.train.data import collate
    a = {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3]),
         "pixel_values_videos": torch.zeros(4, 2), "video_grid_thw": torch.tensor([[1, 2, 2]])}
    b = {"input_ids": torch.tensor([4]), "labels": torch.tensor([4]),
         "pixel_values_videos": torch.zeros(4, 2), "video_grid_thw": torch.tensor([[1, 2, 2]])}
    out = collate([a, b], pad_id=0)
    assert out["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert out["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert out["pixel_values_videos"].shape == (8, 2) and out["video_grid_thw"].shape == (2, 3)


def test_kl_is_zero_for_identical_models():
    from wire.train.loss import DistillConfig, _kl_ce
    torch.manual_seed(0)
    head = nn.Linear(8, 11)
    h = torch.randn(5, 8)
    gt = torch.randint(0, 11, (5,))
    with torch.no_grad():
        kl, ce = _kl_ce(head, h, h, gt, DistillConfig(chunk=2), with_ce=True)
        assert abs(float(kl)) < 1e-5 and float(ce) > 0
        kl, _ = _kl_ce(head, h + 1.0, h, gt, DistillConfig(), with_ce=False)
        assert float(kl) > 0


def test_build_llavavideo_records(tmp_path):
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "data", "build_llavavideo.py")
    spec = importlib.util.spec_from_file_location("build_llavavideo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bucket = tmp_path / "0_30_s_youtube_v0_1"
    (bucket / "videos").mkdir(parents=True)
    (bucket / "videos" / "a.mp4").write_bytes(b"")
    (bucket / "x_qa_processed.json").write_text(json.dumps([
        {"video": "a.mp4", "conversations": [{"value": "<image>\nWhat is shown?"},
                                             {"value": "B. A dog."}]},
        {"video": "missing.mp4", "conversations": [{"value": "Q"}, {"value": "A"}]}]))
    recs = list(mod.records(str(tmp_path)))
    assert len(recs) == 1
    assert recs[0]["messages"][0]["content"] == "<video>What is shown?"
    assert os.path.isabs(recs[0]["videos"][0])
    assert mod.MCQ.match(recs[0]["messages"][1]["content"])
