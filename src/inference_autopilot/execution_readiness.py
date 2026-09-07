"""Conservative launch/defer decisions from audited physical-attempt history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import isfinite, sqrt
from pathlib import Path
from statistics import median
from typing import Any

from inference_autopilot.attempt_ledger import (
    audit_attempt_ledger,
    load_attempt_ledger,
)
from inference_autopilot.calibration.models import (
    CalibrationPlan,
    canonical_json,
    canonical_sha256,
    require_digest,
    require_id,
    require_object,
)


_REPORT_KEYS = {
    "schema_version",
    "producer",
    "readiness_id",
    "status",
    "requirements",
    "source",
    "attempts",
    "summary",
    "decision",
    "execution_readiness_assessment_sha256",
}
_OUTCOMES = {
    "accepted",
    "environment_contaminated",
    "missing_observation",
    "run_failed",
}
_PROBABILITY_MODELS = {"stationary_independent_bernoulli"}


def _expect_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _finite(value: Any, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    parsed = float(value)
    if not isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{context} must be finite and at least {minimum}")
    return parsed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ExecutionReadinessSpec:
    readiness_id: str
    minimum_attempt_count: int
    maximum_environment_retries: int
    wilson_z: float
    minimum_clean_probability_lower_bound: float
    minimum_campaign_completion_probability: float
    maximum_expected_npu_hours: float
    duration_safety_factor: float
    probability_model: str = "stationary_independent_bernoulli"
    reject_non_environment_failures: bool = True
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported execution-readiness spec: {self.schema_version}")
        require_id(self.readiness_id, "readiness_id")
        if (
            isinstance(self.minimum_attempt_count, bool)
            or not isinstance(self.minimum_attempt_count, int)
            or self.minimum_attempt_count <= 0
        ):
            raise ValueError("minimum_attempt_count must be a positive integer")
        if (
            isinstance(self.maximum_environment_retries, bool)
            or not isinstance(self.maximum_environment_retries, int)
            or self.maximum_environment_retries < 0
        ):
            raise ValueError("maximum_environment_retries must be non-negative")
        _finite(self.wilson_z, "wilson_z", minimum=1e-12)
        for name in (
            "minimum_clean_probability_lower_bound",
            "minimum_campaign_completion_probability",
        ):
            value = _finite(getattr(self, name), name)
            if value > 1:
                raise ValueError(f"{name} must be at most one")
        _finite(
            self.maximum_expected_npu_hours,
            "maximum_expected_npu_hours",
            minimum=1e-12,
        )
        _finite(self.duration_safety_factor, "duration_safety_factor", minimum=1.0)
        if self.probability_model not in _PROBABILITY_MODELS:
            raise ValueError(
                f"unsupported execution-readiness probability model: "
                f"{self.probability_model}"
            )
        if not isinstance(self.reject_non_environment_failures, bool):
            raise ValueError("reject_non_environment_failures must be boolean")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "readiness_id": self.readiness_id,
            "minimum_attempt_count": self.minimum_attempt_count,
            "maximum_environment_retries": self.maximum_environment_retries,
            "wilson_z": self.wilson_z,
            "minimum_clean_probability_lower_bound": (
                self.minimum_clean_probability_lower_bound
            ),
            "minimum_campaign_completion_probability": (
                self.minimum_campaign_completion_probability
            ),
            "maximum_expected_npu_hours": self.maximum_expected_npu_hours,
            "duration_safety_factor": self.duration_safety_factor,
            "probability_model": self.probability_model,
            "reject_non_environment_failures": self.reject_non_environment_failures,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExecutionReadinessSpec":
        expected = {
            "schema_version",
            "readiness_id",
            "minimum_attempt_count",
            "maximum_environment_retries",
            "wilson_z",
            "minimum_clean_probability_lower_bound",
            "minimum_campaign_completion_probability",
            "maximum_expected_npu_hours",
            "duration_safety_factor",
            "probability_model",
            "reject_non_environment_failures",
        }
        _expect_keys(raw, expected, "execution-readiness spec")
        return cls(
            schema_version=str(raw["schema_version"]),
            readiness_id=str(raw["readiness_id"]),
            minimum_attempt_count=raw["minimum_attempt_count"],
            maximum_environment_retries=raw["maximum_environment_retries"],
            wilson_z=_finite(raw["wilson_z"], "wilson_z"),
            minimum_clean_probability_lower_bound=_finite(
                raw["minimum_clean_probability_lower_bound"],
                "minimum_clean_probability_lower_bound",
            ),
            minimum_campaign_completion_probability=_finite(
                raw["minimum_campaign_completion_probability"],
                "minimum_campaign_completion_probability",
            ),
            maximum_expected_npu_hours=_finite(
                raw["maximum_expected_npu_hours"], "maximum_expected_npu_hours"
            ),
            duration_safety_factor=_finite(
                raw["duration_safety_factor"], "duration_safety_factor"
            ),
            probability_model=str(raw["probability_model"]),
            reject_non_environment_failures=raw["reject_non_environment_failures"],
        )


def _wilson_lower(successes: int, trials: int, z: float) -> float | None:
    if trials == 0:
        return None
    estimate = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = estimate + z2 / (2.0 * trials)
    radius = z * sqrt(
        estimate * (1.0 - estimate) / trials + z2 / (4.0 * trials * trials)
    )
    return max(0.0, (center - radius) / denominator)


def _derive_summary_and_decision(
    spec: ExecutionReadinessSpec,
    attempts: Sequence[Mapping[str, Any]],
    *,
    planned_run_count: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if planned_run_count <= 0:
        raise ValueError("target campaign must contain a positive run count")
    accepted = sum(attempt["outcome"] == "accepted" for attempt in attempts)
    contaminated = sum(
        attempt["outcome"] == "environment_contaminated" for attempt in attempts
    )
    other_failures = len(attempts) - accepted - contaminated
    classifiable = accepted + contaminated
    point_clean = accepted / classifiable if classifiable else None
    lower_clean = _wilson_lower(accepted, classifiable, spec.wilson_z)
    durations = [
        float(attempt["duration_seconds"])
        for attempt in attempts
        if attempt.get("duration_seconds") is not None
    ]
    median_duration = median(durations) if durations else None
    maximum_attempts = spec.maximum_environment_retries + 1
    if lower_clean is None:
        logical_success = None
        campaign_success = None
        expected_attempts = None
        expected_npu_hours = None
    else:
        logical_success = 1.0 - (1.0 - lower_clean) ** maximum_attempts
        campaign_success = logical_success**planned_run_count
        expected_attempts = sum(
            (1.0 - lower_clean) ** attempt_index
            for attempt_index in range(maximum_attempts)
        )
        expected_npu_hours = (
            None
            if median_duration is None
            else planned_run_count
            * expected_attempts
            * median_duration
            * spec.duration_safety_factor
            / 3600.0
        )

    reasons: list[str] = []
    if classifiable < spec.minimum_attempt_count:
        reasons.append("insufficient_attempt_history")
    if lower_clean is None:
        reasons.append("clean_probability_unavailable")
    elif lower_clean < spec.minimum_clean_probability_lower_bound:
        reasons.append("clean_probability_below_threshold")
    if campaign_success is None:
        reasons.append("campaign_completion_probability_unavailable")
    elif campaign_success < spec.minimum_campaign_completion_probability:
        reasons.append("campaign_completion_probability_below_threshold")
    if median_duration is None:
        reasons.append("attempt_duration_unavailable")
    if expected_npu_hours is not None and (
        expected_npu_hours > spec.maximum_expected_npu_hours
    ):
        reasons.append("expected_npu_budget_exceeded")
    if spec.reject_non_environment_failures and other_failures:
        reasons.append("non_environment_failures_observed")
    reasons = sorted(set(reasons))
    status = "launch" if not reasons else "defer"
    summary = {
        "probability_model": spec.probability_model,
        "planned_run_count": planned_run_count,
        "maximum_attempts_per_logical_run": maximum_attempts,
        "attempt_count": len(attempts),
        "host_classifiable_attempt_count": classifiable,
        "accepted_attempt_count": accepted,
        "environment_contaminated_attempt_count": contaminated,
        "other_failure_attempt_count": other_failures,
        "point_clean_probability": point_clean,
        "wilson_clean_probability_lower_bound": lower_clean,
        "logical_run_completion_probability_lower_bound": logical_success,
        "campaign_completion_probability_lower_bound": campaign_success,
        "median_attempt_duration_seconds": median_duration,
        "duration_sample_count": len(durations),
        "expected_attempts_per_logical_run_conservative_estimate": expected_attempts,
        "expected_npu_hours_conservative_estimate": expected_npu_hours,
    }
    decision = {
        "action": "launch_campaign" if status == "launch" else "defer_host",
        "reasons": reasons,
    }
    return summary, decision, status


def _context_from_plan(plan: CalibrationPlan) -> dict[str, Any]:
    return {
        "semantic_contract": plan.spec.semantic_contract.to_dict(),
        "workload_contract": plan.spec.workload_contract.to_dict(),
        "environment_contract": plan.spec.environment_contract.to_dict(),
    }


def _context_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    workload = require_object(manifest.get("workload_contract"), "workload contract")
    workload.pop("workload_seed", None)
    return {
        "semantic_contract": require_object(
            manifest.get("semantic_contract"), "semantic contract"
        ),
        "workload_contract": workload,
        "environment_contract": require_object(
            manifest.get("environment_contract"), "environment contract"
        ),
    }


def assess_execution_readiness(
    spec: ExecutionReadinessSpec,
    target_plan: CalibrationPlan,
    target_plan_path: Path,
    history_campaign_dirs: Sequence[Path],
) -> "ExecutionReadinessAssessment":
    if not history_campaign_dirs:
        raise ValueError("execution readiness requires at least one history campaign")
    target_context = _context_from_plan(target_plan)
    target_context_sha256 = canonical_sha256(target_context)
    attempts: list[dict[str, Any]] = []
    campaign_sources: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    for campaign_dir in history_campaign_dirs:
        plan_path = campaign_dir / "plan.json"
        ledger_path = campaign_dir / "attempt-ledger.jsonl"
        audit = audit_attempt_ledger(
            ledger_path,
            campaign_dir=campaign_dir,
            plan_path=plan_path,
            require_complete=False,
        )
        history_plan = CalibrationPlan.from_dict(
            require_object(
                json.loads(plan_path.read_text(encoding="utf-8")),
                "history calibration plan",
            )
        )
        if _context_from_plan(history_plan) != target_context:
            raise ValueError(f"history campaign context mismatch: {campaign_dir}")
        records = load_attempt_ledger(ledger_path)
        for record in records:
            record_digest = str(record["record_sha256"])
            if record_digest in seen_records:
                raise ValueError("duplicate physical attempt across history campaigns")
            seen_records.add(record_digest)
            bundle_dir = campaign_dir / str(record["bundle_path"])
            manifest = require_object(
                json.loads((bundle_dir / "run-manifest.json").read_text(encoding="utf-8")),
                "attempt run manifest",
            )
            if _context_from_manifest(manifest) != target_context:
                raise ValueError("attempt manifest context differs from target campaign")
            report_binding = record.get("host_interference_report")
            duration: float | None = None
            if isinstance(report_binding, Mapping):
                report_path = campaign_dir / str(report_binding["path"])
                report = require_object(
                    json.loads(report_path.read_text(encoding="utf-8")),
                    "host interference report",
                )
                summary = report.get("summary")
                raw_duration = summary.get("duration_seconds") if isinstance(summary, Mapping) else None
                if isinstance(raw_duration, (int, float)) and not isinstance(raw_duration, bool):
                    duration = _finite(raw_duration, "attempt duration")
            outcome = str(record["outcome"])
            if outcome not in _OUTCOMES:
                raise ValueError(f"unsupported attempt outcome: {outcome}")
            attempts.append(
                {
                    "campaign_id": history_plan.spec.campaign_id,
                    "logical_run_id": record["logical_run_id"],
                    "attempt_index": record["attempt_index"],
                    "outcome": outcome,
                    "duration_seconds": duration,
                    "record_sha256": record_digest,
                }
            )
        campaign_sources.append(
            {
                "campaign_id": history_plan.spec.campaign_id,
                "campaign_plan_file_sha256": _file_sha256(plan_path),
                "attempt_ledger_file_sha256": _file_sha256(ledger_path),
                "attempt_ledger_final_record_sha256": audit["final_record_sha256"],
                "attempt_ledger_status": audit["status"],
                "attempt_count": audit["attempt_count"],
            }
        )
    summary, decision, status = _derive_summary_and_decision(
        spec,
        attempts,
        planned_run_count=len(target_plan.runs),
    )
    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "readiness_id": spec.readiness_id,
        "status": status,
        "requirements": spec.to_dict(),
        "source": {
            "requirements_sha256": spec.sha256,
            "target_plan_file_sha256": _file_sha256(target_plan_path),
            "target_plan_sha256": canonical_sha256(target_plan.to_dict()),
            "target_context_sha256": target_context_sha256,
            "history_campaigns": campaign_sources,
        },
        "attempts": attempts,
        "summary": summary,
        "decision": decision,
    }
    return ExecutionReadinessAssessment(
        {**payload, "execution_readiness_assessment_sha256": canonical_sha256(payload)}
    )


@dataclass(frozen=True, slots=True)
class ExecutionReadinessAssessment:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _expect_keys(self.payload, _REPORT_KEYS, "execution-readiness assessment")
        raw = dict(self.payload)
        digest = raw.pop("execution_readiness_assessment_sha256", None)
        require_digest(str(digest), "execution_readiness_assessment_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("execution-readiness assessment SHA256 mismatch")
        if raw["schema_version"] != "1.0" or raw["producer"] != "inference-autopilot/0.1.0":
            raise ValueError("unsupported execution-readiness assessment")
        spec = ExecutionReadinessSpec.from_dict(
            require_object(raw["requirements"], "execution-readiness requirements")
        )
        if raw["readiness_id"] != spec.readiness_id:
            raise ValueError("execution-readiness id does not match requirements")
        source = require_object(raw["source"], "execution-readiness source")
        _expect_keys(
            source,
            {
                "requirements_sha256",
                "target_plan_file_sha256",
                "target_plan_sha256",
                "target_context_sha256",
                "history_campaigns",
            },
            "execution-readiness source",
        )
        if source.get("requirements_sha256") != spec.sha256:
            raise ValueError("execution-readiness requirements digest mismatch")
        for name in (
            "target_plan_file_sha256",
            "target_plan_sha256",
            "target_context_sha256",
        ):
            require_digest(str(source.get(name)), name)
        attempts = raw["attempts"]
        if not isinstance(attempts, list) or any(not isinstance(item, Mapping) for item in attempts):
            raise ValueError("execution-readiness attempts must be an array of objects")
        record_digests: set[str] = set()
        for attempt in attempts:
            _expect_keys(
                attempt,
                {
                    "campaign_id",
                    "logical_run_id",
                    "attempt_index",
                    "outcome",
                    "duration_seconds",
                    "record_sha256",
                },
                "execution-readiness attempt",
            )
            require_id(str(attempt["campaign_id"]), "attempt campaign_id")
            require_id(str(attempt["logical_run_id"]), "attempt logical_run_id")
            require_digest(str(attempt["record_sha256"]), "attempt record_sha256")
            if attempt["record_sha256"] in record_digests:
                raise ValueError("execution-readiness attempt records must be unique")
            record_digests.add(str(attempt["record_sha256"]))
            if attempt["outcome"] not in _OUTCOMES:
                raise ValueError("execution-readiness attempt outcome is invalid")
            if attempt["duration_seconds"] is not None:
                _finite(attempt["duration_seconds"], "attempt duration")
        history_campaigns = source["history_campaigns"]
        if not isinstance(history_campaigns, list) or any(
            not isinstance(item, Mapping) for item in history_campaigns
        ):
            raise ValueError("history_campaigns must be an array of objects")
        source_campaign_ids: list[str] = []
        for campaign in history_campaigns:
            _expect_keys(
                campaign,
                {
                    "campaign_id",
                    "campaign_plan_file_sha256",
                    "attempt_ledger_file_sha256",
                    "attempt_ledger_final_record_sha256",
                    "attempt_ledger_status",
                    "attempt_count",
                },
                "execution-readiness history campaign",
            )
            campaign_id = str(campaign["campaign_id"])
            require_id(campaign_id, "history campaign_id")
            source_campaign_ids.append(campaign_id)
            for name in (
                "campaign_plan_file_sha256",
                "attempt_ledger_file_sha256",
            ):
                require_digest(str(campaign[name]), name)
            final_record = campaign["attempt_ledger_final_record_sha256"]
            if final_record is not None:
                require_digest(str(final_record), "attempt_ledger_final_record_sha256")
            if campaign["attempt_ledger_status"] not in {"complete", "partial"}:
                raise ValueError("attempt ledger status must be complete or partial")
            attempt_count = campaign["attempt_count"]
            if (
                isinstance(attempt_count, bool)
                or not isinstance(attempt_count, int)
                or attempt_count < 0
            ):
                raise ValueError("history campaign attempt_count must be non-negative")
            observed_count = sum(
                attempt["campaign_id"] == campaign_id for attempt in attempts
            )
            if attempt_count != observed_count:
                raise ValueError("history campaign attempt count does not match attempts")
        if len(source_campaign_ids) != len(set(source_campaign_ids)):
            raise ValueError("history campaign ids must be unique")
        if {str(attempt["campaign_id"]) for attempt in attempts} - set(source_campaign_ids):
            raise ValueError("attempt references an undeclared history campaign")
        summary = require_object(raw["summary"], "execution-readiness summary")
        planned_run_count = summary.get("planned_run_count")
        if isinstance(planned_run_count, bool) or not isinstance(planned_run_count, int):
            raise ValueError("planned_run_count must be an integer")
        expected_summary, expected_decision, expected_status = _derive_summary_and_decision(
            spec,
            attempts,
            planned_run_count=planned_run_count,
        )
        if summary != expected_summary:
            raise ValueError("execution-readiness summary does not match attempts")
        if raw["decision"] != expected_decision or raw["status"] != expected_status:
            raise ValueError("execution-readiness decision does not match evidence")
        canonical = canonical_json(self.payload)
        object.__setattr__(self, "payload", json.loads(canonical))

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.payload))

    def audit(self) -> dict[str, Any]:
        return {
            "readiness_id": self.payload["readiness_id"],
            "status": self.status,
            "attempt_count": self.payload["summary"]["attempt_count"],
            "campaign_completion_probability_lower_bound": self.payload["summary"][
                "campaign_completion_probability_lower_bound"
            ],
            "expected_npu_hours_conservative_estimate": self.payload["summary"][
                "expected_npu_hours_conservative_estimate"
            ],
            "action": self.payload["decision"]["action"],
            "reasons": list(self.payload["decision"]["reasons"]),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExecutionReadinessAssessment":
        return cls(raw)
