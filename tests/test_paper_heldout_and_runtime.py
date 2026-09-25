"""CPU-only tests for paper transfer and overhead result aggregation."""

import math

import torch

from ops import paper_heldout_mc3, paper_runtime
from ops.codec_search_ar import QPS
from ops.paper_heldout_mc3 import curves
from ops.paper_heldout_mc3 import summarize as heldout_summary
from ops.paper_runtime import summarize as runtime_summary


def _heldout_records(n=40):
    rows = []
    for i in range(n):
        measurements = []
        for position, qp in enumerate(QPS):
            rate = .5 * (.72 ** position)
            measurements.append({"qp": qp,
                "anchor": {"bpp": rate, "correct": i < (34 - 5 * position),
                           "encode_decode_s": .1, "inference_s": .02},
                "trial": {"bpp": rate * .9, "correct": i < (35 - 5 * position),
                          "encode_decode_s": .1, "inference_s": .02}})
        rows.append({"sequence_id": f"clip-{i}", "measurements": measurements})
    return rows


def test_heldout_uses_video_as_bootstrap_unit():
    records = _heldout_records()
    anchor, trial = curves(records)
    assert all(anchor[str(qp)]["n"] == 40 for qp in QPS)
    assert trial[str(QPS[0])]["bpp"] < anchor[str(QPS[0])]["bpp"]
    report = heldout_summary(records, draws=12)
    assert report["n"] == 40
    assert math.isfinite(report["metrics"]["bd_rate_top1_pct"])
    assert report["bootstrap"]["bd_rate_top1_pct"]["requested_draws"] == 12
    assert report["timing"]["anchor_encode_decode_s"]["n"] == 40 * len(QPS)


def test_runtime_summary_reports_per_clip_overhead():
    report = runtime_summary([
        {"qps": [{"qp": 40}], "identity_only_s": 1., "full_selector_s": 6.,
         "overhead_ratio": 6., "candidate_codec_s": 3., "analyzer_s": 2.,
         "selector_s": .01},
        {"qps": [{"qp": 40}], "identity_only_s": 2., "full_selector_s": 10.,
         "overhead_ratio": 5., "candidate_codec_s": 5., "analyzer_s": 3.,
         "selector_s": .02},
    ])
    assert report["n"] == 2
    assert report["qps"] == [40]
    assert report["baseline_codec_calls_per_clip"] == 1
    assert report["full_codec_calls_per_clip"] == 6
    assert report["overhead_ratio"]["median"] == 5.5
    assert report["full_selector_s"]["mean"] == 8.


class _FakeDataset:
    def __init__(self):
        self.samples = [{"path": "fake/class_clip.mp4"}]

    def __getitem__(self, _index):
        return (torch.ones(3, 16, 128, 128) * .5, 1,
                {"sequence_id": "class/clip.mp4"})


class _FakeCodec:
    def __init__(self):
        self.calls = []

    def _encode_decode_clip(self, candidate, qp):
        self.calls.append(qp)
        return candidate, .3


def test_mc3_runner_keeps_frozen_choice(monkeypatch):
    class Capture:
        def read(self):
            return True, None

        def release(self):
            pass

    monkeypatch.setattr(paper_heldout_mc3.cv2, "VideoCapture", lambda _path: Capture())
    monkeypatch.setattr(paper_heldout_mc3, "select", lambda *_args: 1)
    monkeypatch.setattr(paper_heldout_mc3, "timed_inference",
                        lambda *_args: (1, .01))
    cache = {"sequence_id": "class/clip.mp4", "measurements": [
        {"qp": qp, "candidates": [{"name": "identity128", "bpp": .3},
                                   {"name": "area112", "bpp": .2}]}
        for qp in QPS]}
    row = paper_heldout_mc3.evaluate_clip(
        _FakeDataset(), 0, cache, {}, {}, object(), _FakeCodec())
    assert len(row["measurements"]) == 5
    assert all(m["chosen"] == "area112" for m in row["measurements"])
    assert all(m["trial"]["correct"] for m in row["measurements"])


def test_full_runtime_includes_all_six_candidates(monkeypatch):
    logits = torch.tensor([[.1, .9]], dtype=torch.float32)
    feature = torch.ones(1, 4)
    monkeypatch.setattr(paper_runtime, "measured_prediction",
                        lambda *_args: ((logits, feature), .001))
    monkeypatch.setattr(paper_runtime, "select", lambda *_args: 0)
    analyzers = {model: object() for model in paper_runtime.MODELS}
    codec = _FakeCodec()
    row = paper_runtime.measure_clip(
        _FakeDataset(), 0, analyzers, codec, {}, {})
    assert len(row["qps"]) == 5
    assert all(len(q["candidate_costs"]) == 6 for q in row["qps"])
    assert row["baseline_codec_calls"] == 5
    assert row["full_codec_calls"] == 30
    assert row["baseline_analyzer_calls"] == 0
    assert row["full_analyzer_calls"] == 62
    assert len(codec.calls) == 35
    assert row["identity_only_s"] > 0
    assert row["full_selector_s"] > 0
    assert row["arm_order"] in (["full", "identity"], ["identity", "full"])


def test_runtime_one_qp_measures_six_versus_one(monkeypatch):
    logits = torch.tensor([[.1, .9]], dtype=torch.float32)
    feature = torch.ones(1, 4)
    monkeypatch.setattr(paper_runtime, "measured_prediction",
                        lambda *_args: ((logits, feature), .001))
    monkeypatch.setattr(paper_runtime, "select", lambda *_args: 0)
    codec = _FakeCodec()
    row = paper_runtime.measure_clip(
        _FakeDataset(), 0, {model: object() for model in paper_runtime.MODELS},
        codec, {}, {}, qps=(40,))
    assert [point["qp"] for point in row["qps"]] == [40]
    assert row["baseline_codec_calls"] == 1
    assert row["full_codec_calls"] == 6
    assert row["full_analyzer_calls"] == 14
    assert len(codec.calls) == 7
