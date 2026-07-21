import pytest

from training.report_wall_clock import compare_reports, summarize_report


def _payload(*, async_enabled=True, total=120.0, grouped=True):
    return {
        "total_wall_seconds": total,
        "dataset_builds": [{"label": "train", "seconds": 5.0}],
        "phases": [
            {
                "name": "Phase1",
                "start_step": 0,
                "end_step": 10,
                "wall_seconds": 100.0,
                "data_wait_seconds": 4.0,
                "optimizer_seconds": 6.0,
                "tokens_processed": 1000,
            }
        ],
        "checkpoint_serializations": [
            {"name": "step.pt", "phase": "Phase1", "seconds": 3.0},
            {"name": "final.pt", "phase": "finance_final", "seconds": 2.0},
        ],
        "async_checkpoint_copy": {
            "enabled": async_enabled,
            "files_copied": 2 if async_enabled else 0,
            "bytes_copied": 20 * 1024**2 if async_enabled else 0,
            "copy_seconds": 2.0 if async_enabled else 0.0,
            "max_queue_depth": 1 if async_enabled else 0,
            "pending": 0,
            "errors": [],
        },
        "runtime": {
            "dtype": "torch.bfloat16",
            "compile": True,
            "grouped_moe": grouped,
            "liger_fused_ce": False,
            "optimizer_backend": "torch_adamw_fused",
        },
    }


def test_summarize_report_separates_foreground_and_background_io():
    summary = summarize_report(_payload(), label="async")
    assert summary["effective_tokens_per_second"] == 10.0
    assert summary["setup_and_finalize_seconds"] == 20.0
    assert summary["checkpoint_serialize_seconds"] == 5.0
    assert summary["phase_compute_and_other_seconds"] == 87.0
    assert summary["async_copy"]["copy_seconds"] == 2.0
    assert summary["async_copy"]["copy_mib_per_second"] == 10.0
    assert summary["phases"] == [
        {
            "name": "Phase1",
            "wall_seconds": 100.0,
            "tokens_processed": 1000,
            "effective_tokens_per_second": 10.0,
            "data_wait_seconds": 4.0,
            "optimizer_seconds": 6.0,
            "checkpoint_serialize_seconds": 3.0,
            "compute_and_other_seconds": 87.0,
        }
    ]


def test_compare_reports_calculates_wall_clock_speedup():
    baseline = summarize_report(
        _payload(async_enabled=False, total=120.0), label="direct"
    )
    candidate = summarize_report(_payload(total=100.0), label="async")
    comparison = compare_reports(baseline, candidate)
    assert comparison["wall_clock_speedup"] == pytest.approx(1.2)
    assert comparison["total_wall_seconds_delta"] == -20.0


def test_compare_reports_rejects_runtime_mismatch():
    baseline = summarize_report(_payload(), label="baseline")
    candidate = summarize_report(
        _payload(grouped=False), label="different-runtime"
    )
    with pytest.raises(ValueError, match="runtime.grouped_moe"):
        compare_reports(baseline, candidate)


def test_empty_phase_report_is_labeled_as_no_training_work():
    payload = _payload()
    payload["phases"] = []
    summary = summarize_report(payload, label="skipped")
    assert summary["training_work_performed"] is False
    assert summary["tokens_processed"] == 0
    assert summary["effective_tokens_per_second"] == 0.0

    baseline = summarize_report(_payload(), label="trained")
    with pytest.raises(ValueError, match="contains no training phases"):
        compare_reports(baseline, summary)


@pytest.mark.parametrize(
    ("pending", "errors"),
    [(1, []), (0, ["copy failed"])],
)
def test_summarize_report_rejects_incomplete_async_copy(pending, errors):
    payload = _payload()
    payload["async_checkpoint_copy"]["pending"] = pending
    payload["async_checkpoint_copy"]["errors"] = errors
    with pytest.raises(ValueError, match="asynchronous copy incomplete"):
        summarize_report(payload, label="bad")
