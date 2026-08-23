from training.rival_training.action_parity import (
    _temporal_window,
    build_action_parity_report,
)


def test_spatial_action_parity_is_exact() -> None:
    report = build_action_parity_report()
    spatial = report["spatial_controller_parity"]
    assert report["status"] == "passed"
    assert spatial["exact_row_case_comparisons"] == 360
    assert spatial["maximum_abs_controller_error"] == 0.0
    assert spatial["appended_index_90_rejected_by_legacy_only_deployment"] is True


def test_temporal_windows_separate_four_and_eight_tick_behavior() -> None:
    tick4 = _temporal_window(4)
    tick8 = _temporal_window(8)
    assert tick4["exact_window_match"] is True
    assert tick4["mismatch_ticks"] == []
    assert tick8["exact_window_match"] is False
    assert tick8["mismatch_ticks"] == [2, 3, 4, 5]
