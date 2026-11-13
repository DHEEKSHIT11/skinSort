from datetime import datetime, timedelta, timezone

import pytest

from src.queue.ranker import (
    AppointmentRequest,
    QueueRanker,
    RankingConfig,
    SeverityPrediction,
)


NOW = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)


def request(case_id, severity, days_waited=0, **kwargs):
    return AppointmentRequest(
        case_id=case_id,
        submitted_at=NOW - timedelta(days=days_waited),
        severity=SeverityPrediction(**severity),
        **kwargs,
    )


def test_safety_threshold_keeps_urgent_case_first():
    ranker = QueueRanker()
    cases = [
        request("routine-old", {"routine": 0.9, "priority": 0.08, "urgent": 0.02}, 365),
        request("urgent-new", {"routine": 0.1, "priority": 0.1, "urgent": 0.8}),
    ]

    ranked = ranker.rank_many(cases, NOW)

    assert [case.case_id for case in ranked] == ["urgent-new", "routine-old"]
    assert ranked[0].safety_override is True


def test_wait_time_changes_order_within_same_severity():
    ranker = QueueRanker()
    cases = [
        request("new", {"routine": 0.8, "priority": 0.15, "urgent": 0.05}),
        request("old", {"routine": 0.8, "priority": 0.15, "urgent": 0.05}, 30),
    ]

    ranked = ranker.rank_many(cases, NOW)

    assert [case.case_id for case in ranked] == ["old", "new"]
    assert ranked[0].wait_component > ranked[1].wait_component


def test_closing_appointment_window_adds_priority():
    ranker = QueueRanker(RankingConfig(deadline_weight=100))
    closing = request(
        "closing",
        {"routine": 0.8, "priority": 0.15, "urgent": 0.05},
        latest_at=NOW + timedelta(hours=1),
    )
    flexible = request(
        "flexible",
        {"routine": 0.8, "priority": 0.15, "urgent": 0.05},
        latest_at=NOW + timedelta(days=30),
    )

    ranked = ranker.rank_many([flexible, closing], NOW)

    assert [case.case_id for case in ranked] == ["closing", "flexible"]
    assert ranked[0].window_status == "open"
    assert ranked[0].deadline_component > ranked[1].deadline_component


def test_future_availability_is_not_eligible_now():
    ranker = QueueRanker()
    future = request(
        "future",
        {"routine": 0.1, "priority": 0.1, "urgent": 0.8},
        earliest_at=NOW + timedelta(days=2),
    )

    result = ranker.rank(future, NOW)

    assert result.window_status == "not_available"
    assert result.eligible_now is False


def test_invalid_probabilities_are_rejected():
    case = request("bad", {"routine": 0.5, "priority": 0.5, "urgent": 0.5})

    with pytest.raises(ValueError, match="sum to 1"):
        QueueRanker().rank(case, NOW)


def test_clinician_override_is_explainable_and_strong():
    ranker = QueueRanker()
    normal = request("normal", {"routine": 0.8, "priority": 0.15, "urgent": 0.05}, 60)
    override = request(
        "override",
        {"routine": 0.8, "priority": 0.15, "urgent": 0.05},
        clinician_override=1,
    )

    ranked = ranker.rank_many([normal, override], NOW)

    assert ranked[0].case_id == "override"
    assert ranked[0].clinician_override == 1
