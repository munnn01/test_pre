"""Pure, fast checks for V8 reporting and commit-pinned Kaggle packaging."""
from __future__ import annotations

from ops.codec_search_ar import QPS
from ops.push_v8_eval import payload
from ops.v8_eval import ARMS, MODELS, compare_curves, report_curves, summarize


def records(n: int = 20) -> list[dict]:
    rows = []
    for i in range(n):
        measurements = []
        for q, qp in enumerate(QPS):
            arms = {}
            for arm in ARMS:
                arms[arm] = {
                    "bpp": 0.5 - 0.08 * q - (0.02 if arm.startswith("v2") else 0),
                    "correct": {model: i < (5 + 2 * q +
                               (1 if arm.endswith("restored") else 0))
                                for model in MODELS},
                    "restoration_s": 0.1 if arm.endswith("restored") else 0.0,
                    "encode_decode_s": 0.2,
                    "analyzer_s": 0.3,
                }
            measurements.append({"qp": qp, "arms": arms})
        rows.append({"sequence_id": str(i), "measurements": measurements})
    return rows


def test_curves_keep_qp_pairing_and_three_models():
    curves = report_curves(records())
    assert set(curves) == set(MODELS)
    assert all(list(curves[model]["codec"]) == [str(q) for q in QPS]
               for model in MODELS)
    report = summarize(records(), 0)
    assert report["n"] == 20
    assert abs(report["runtime"]["v2_restored"]["mean_restoration_s_per_16_frames"] - 0.1) < 1e-9


def test_undefined_bd_rate_is_json_safe():
    curves = report_curves(records())
    for model in MODELS:
        for arm in ARMS:
            for qp in QPS:
                curves[model][arm][str(qp)]["top1"] = 0.5
    result = compare_curves(curves)
    assert all(cell["bd_rate_top1_pct"] is None
               for model in result.values() for cell in model.values())


def test_eval_payload_uses_private_sources_and_locked_commit():
    commit = "a" * 40
    book, meta = payload(commit, "qktttttttttt", 1, "v8-restorer-h264-" + "b" * 12)
    source = "".join(book["cells"][0]["source"])
    assert commit in source and "__REF__" not in source
    assert meta["is_private"] and len(meta["dataset_sources"]) == 3
    assert meta["id"].endswith("-s1")
