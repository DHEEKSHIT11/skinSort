"""Severity-aware queue ranking with wait-time and appointment-window features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping


SEVERITY_LEVELS = ("routine", "priority", "urgent")
SEVERITY_WEIGHTS = {"routine": 10.0, "priority": 50.0, "urgent": 100.0}


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class SeverityPrediction:
    """Output contract for the image-severity model."""

    routine: float
    priority: float
    urgent: float
    confidence: float | None = None
    quality_ok: bool = True

    def probabilities(self) -> Mapping[str, float]:
        return {
            "routine": self.routine,
            "priority": self.priority,
            "urgent": self.urgent,
        }

    def validate(self) -> None:
        probabilities = self.probabilities()
        if any(not math.isfinite(value) or value < 0 or value > 1
               for value in probabilities.values()):
            raise ValueError("Severity probabilities must be finite values between 0 and 1")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
            raise ValueError("Severity probabilities must sum to 1")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass(frozen=True)
class AppointmentRequest:
    """A case waiting for a dermatologist appointment."""

    case_id: str
    submitted_at: datetime
    severity: SeverityPrediction
    earliest_at: datetime | None = None
    latest_at: datetime | None = None
    clinician_override: int = 0

    def validate(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if self.earliest_at and self.latest_at:
            if _as_utc(self.earliest_at) > _as_utc(self.latest_at):
                raise ValueError("earliest_at must not be after latest_at")
        if not isinstance(self.clinician_override, int):
            raise ValueError("clinician_override must be an integer")
        self.severity.validate()


@dataclass(frozen=True)
class RankingConfig:
    """Tunable ranking parameters for the first queue baseline."""

    urgent_threshold: float = 0.70
    wait_weight: float = 8.0
    deadline_weight: float = 25.0
    deadline_horizon: timedelta = timedelta(hours=72)

    def validate(self) -> None:
        if not 0 < self.urgent_threshold <= 1:
            raise ValueError("urgent_threshold must be between 0 and 1")
        if self.wait_weight < 0 or self.deadline_weight < 0:
            raise ValueError("Ranking weights must not be negative")
        if self.deadline_horizon.total_seconds() <= 0:
            raise ValueError("deadline_horizon must be positive")


@dataclass(frozen=True)
class RankedCase:
    """A queue result with explainable feature contributions."""

    case_id: str
    score: float
    severity_tier: str
    wait_hours: float
    window_status: str
    eligible_now: bool
    safety_override: bool
    clinician_override: int
    severity_component: float
    wait_component: float
    deadline_component: float


class QueueRanker:
    """Rank appointments from severity predictions and scheduling context.

    The ranker is intentionally transparent until clinician-approved ranking
    labels exist for training a learned second-stage model. Waiting time grows
    logarithmically, while a closing or missed appointment window adds a
    separate nonlinear component. Cases above the urgent safety threshold are
    always placed ahead of non-overridden cases.
    """

    def __init__(self, config: RankingConfig | None = None):
        self.config = config or RankingConfig()
        self.config.validate()

    def rank(self, request: AppointmentRequest, now: datetime | None = None) -> RankedCase:
        request.validate()
        current = _as_utc(now or datetime.now(timezone.utc))
        submitted = _as_utc(request.submitted_at)
        wait_hours = max((current - submitted).total_seconds() / 3600, 0.0)

        probabilities = request.severity.probabilities()
        predicted_tier = max(probabilities, key=probabilities.get)
        safety_override = probabilities["urgent"] >= self.config.urgent_threshold
        severity_tier = "urgent" if safety_override else predicted_tier

        window_status, eligible_now = self._window_status(request, current)
        severity_component = sum(
            probabilities[level] * SEVERITY_WEIGHTS[level]
            for level in SEVERITY_LEVELS
        )
        wait_component = self.config.wait_weight * math.log1p(wait_hours)
        deadline_component = self._deadline_component(request, current)
        override_component = request.clinician_override * 1000.0
        safety_component = 10000.0 if safety_override else 0.0

        return RankedCase(
            case_id=request.case_id,
            score=(severity_component + wait_component + deadline_component
                   + override_component + safety_component),
            severity_tier=severity_tier,
            wait_hours=wait_hours,
            window_status=window_status,
            eligible_now=eligible_now,
            safety_override=safety_override,
            clinician_override=request.clinician_override,
            severity_component=severity_component,
            wait_component=wait_component,
            deadline_component=deadline_component,
        )

    def rank_many(
        self,
        requests: Iterable[AppointmentRequest],
        now: datetime | None = None,
    ) -> list[RankedCase]:
        current = _as_utc(now or datetime.now(timezone.utc))
        ranked = [self.rank(request, current) for request in requests]
        return sorted(
            ranked,
            key=lambda result: (
                not result.eligible_now,
                not result.safety_override,
                -result.clinician_override,
                -result.score,
                -result.wait_hours,
                result.case_id,
            ),
        )

    @staticmethod
    def _window_status(
        request: AppointmentRequest,
        now: datetime,
    ) -> tuple[str, bool]:
        earliest = _as_utc(request.earliest_at) if request.earliest_at else None
        latest = _as_utc(request.latest_at) if request.latest_at else None
        if earliest and now < earliest:
            return "not_available", False
        if latest and now > latest:
            return "overdue", True
        return "open", True

    def _deadline_component(
        self,
        request: AppointmentRequest,
        now: datetime,
    ) -> float:
        if request.latest_at is None:
            return 0.0
        latest = _as_utc(request.latest_at)
        remaining = (latest - now).total_seconds()
        horizon = self.config.deadline_horizon.total_seconds()
        urgency = 1.0 if remaining <= 0 else max(0.0, 1.0 - remaining / horizon)
        return self.config.deadline_weight * urgency
