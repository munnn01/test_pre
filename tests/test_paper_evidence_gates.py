"""Keep the paper gate decisions tied to committed result artifacts."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_predeclared_dual_analyzer_target_is_not_relabelled_as_pass():
    base = "results/dual_codec_search_v2_confirm_1000"
    h264 = _read(f"{base}/h264_result.json")
    h265 = _read(f"{base}/h265_result.json")
    assert h264["point_target_met"] is False
    assert h265["point_target_met"] is False
    observed = h264["arms"]["dual_v2"]["analyzers"]["r3d_18"]["metrics"][
        "bd_rate_top1_pct"]
    assert -15 < observed < -14


def test_revised_directional_objective_is_descriptive_on_existing_results():
    base = "results/dual_codec_search_v2_confirm_1000"
    for codec in ("h264", "h265"):
        report = _read(f"{base}/{codec}_result.json")
        for model in ("r2plus1d_18", "r3d_18"):
            metrics = report["arms"]["dual_v2"]["analyzers"][model]["metrics"]
            assert metrics["bd_rate_top1_pct"] < 0
            assert metrics["bd_accuracy_top1_pp"] > 0


def test_runtime_is_measured_pilot_not_unmeasured_cost():
    base = "results/paper_runtime_v2_qp40_20"
    for codec in ("h264", "h265"):
        result = _read(f"{base}/{codec}_result.json")
        assert result["manifest"]["codec"] == codec
        assert result["summary"]["n"] == 20
        assert result["summary"]["qps"] == [40]
        assert len(result["records"]) == 20
        assert result["summary"]["baseline_codec_calls_per_clip"] == 1
        assert result["summary"]["full_codec_calls_per_clip"] == 6
        assert result["summary"]["overhead_ratio"]["median"] > 6


def test_od_pilot_is_distinct_and_h265_uncertainty_crosses_zero():
    result = _read("results/paper_od_pilot_100/probe_bgsuppress.json")
    assert result["n_images"] == 100
    assert result["eval_backbone"] != result["mask_backbone"]
    interval = result["bootstrap_ci"]["h265"]["blur4"]
    assert interval["lo"] < 0 < interval["hi"]
    assert interval["n_draws"] == 100
