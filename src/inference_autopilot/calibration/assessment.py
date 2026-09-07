"""Formal gates that turn completed paired runs into graded evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from math import exp, isfinite, log
from pathlib import Path
from statistics import median
from typing import Any

from inference_autopilot.calibration.models import (
    CalibrationPlan,
    MetricConstraint,
    PlannedRun,
    RunObservation,
    canonical_sha256,
    require_digest,
    require_object,
)
from inference_autopilot.calibration.planning import build_run_manifest
from inference_autopilot.evidence import (
    EvidenceGrade,
    EvidenceLedger,
    EvidencePurpose,
    EvidenceRecord,
    QualityAssessment,
    SourceArtifact,
)


@dataclass(frozen=True, slots=True)
class LoadedObservation:
    observation: RunObservation
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class AssessmentIssue:
    code: str
    message: str
    comparison_group_id: str | None = None
    run_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "comparison_group_id": self.comparison_group_id,
            "run_ids": list(self.run_ids),
        }


@dataclass(frozen=True, slots=True)
class PairedMetricEffect:
    pair_index: int
    workload_seed: int
    baseline_run_id: str
    candidate_run_id: str
    baseline_value: float
    candidate_value: float
    candidate_over_baseline_ratio: float
    directional_relative_improvement: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_index": self.pair_index,
            "workload_seed": self.workload_seed,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
            "candidate_over_baseline_ratio": self.candidate_over_baseline_ratio,
            "directional_relative_improvement": (
                self.directional_relative_improvement
            ),
        }


@dataclass(frozen=True, slots=True)
class ComparisonEffect:
    comparison_group_id: str
    candidate_configuration_id: str
    primary_metric: str
    direction: str
    replay_control: bool
    formal_group: bool
    quality_constraints_satisfied: bool
    expected_pair_count: int
    complete_pair_count: int
    pair_effects: tuple[PairedMetricEffect, ...]
    candidate_over_baseline_geomean_ratio: float | None
    median_directional_relative_improvement: float | None
    effect_exceeds_replay_noise: bool | None = None
    effect_outside_replay_noise: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_group_id": self.comparison_group_id,
            "candidate_configuration_id": self.candidate_configuration_id,
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "replay_control": self.replay_control,
            "formal_group": self.formal_group,
            "quality_constraints_satisfied": self.quality_constraints_satisfied,
            "expected_pair_count": self.expected_pair_count,
            "complete_pair_count": self.complete_pair_count,
            "pair_effects": [effect.to_dict() for effect in self.pair_effects],
            "candidate_over_baseline_geomean_ratio": (
                self.candidate_over_baseline_geomean_ratio
            ),
            "median_directional_relative_improvement": (
                self.median_directional_relative_improvement
            ),
            "effect_exceeds_replay_noise": self.effect_exceeds_replay_noise,
            "effect_outside_replay_noise": self.effect_outside_replay_noise,
        }


@dataclass(frozen=True, slots=True)
class ReplayNoiseReference:
    assessment_sha256: str
    plan_sha256: str
    primary_metric: str
    direction: str
    replay_noise_envelope: float
    comparison_group_ids: tuple[str, ...]
    algorithm_context_sha256: str
    workload_context_sha256: str
    environment_context_sha256: str
    baseline_settings_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "assessment_sha256",
            "plan_sha256",
            "algorithm_context_sha256",
            "workload_context_sha256",
            "environment_context_sha256",
            "baseline_settings_sha256",
        ):
            require_digest(getattr(self, name), name)
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("replay-noise direction must be maximize or minimize")
        if (
            not isfinite(self.replay_noise_envelope)
            or self.replay_noise_envelope < 0
        ):
            raise ValueError("replay-noise envelope must be finite and non-negative")
        if (
            not self.comparison_group_ids
            or tuple(sorted(set(self.comparison_group_ids)))
            != self.comparison_group_ids
        ):
            raise ValueError("replay-noise comparison groups must be sorted and unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_sha256": self.assessment_sha256,
            "plan_sha256": self.plan_sha256,
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "replay_noise_envelope": self.replay_noise_envelope,
            "comparison_group_ids": list(self.comparison_group_ids),
            "algorithm_context_sha256": self.algorithm_context_sha256,
            "workload_context_sha256": self.workload_context_sha256,
            "environment_context_sha256": self.environment_context_sha256,
            "baseline_settings_sha256": self.baseline_settings_sha256,
        }


@dataclass(frozen=True, slots=True)
class CalibrationAssessment:
    plan_sha256: str
    total_group_count: int
    formal_group_count: int
    issues: tuple[AssessmentIssue, ...]
    effects: tuple[ComparisonEffect, ...]
    local_replay_noise_envelope: float | None
    replay_noise_envelope: float | None
    replay_noise_reference: ReplayNoiseReference | None
    ledger: EvidenceLedger
    schema_version: str = "1.0"

    @property
    def formal_complete(self) -> bool:
        return self.formal_group_count == self.total_group_count and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "total_group_count": self.total_group_count,
            "formal_group_count": self.formal_group_count,
            "formal_complete": self.formal_complete,
            "issues": [issue.to_dict() for issue in self.issues],
            "effects": [effect.to_dict() for effect in self.effects],
            "local_replay_noise_envelope": self.local_replay_noise_envelope,
            "replay_noise_envelope": self.replay_noise_envelope,
            "replay_noise_reference": (
                None
                if self.replay_noise_reference is None
                else self.replay_noise_reference.to_dict()
            ),
            "ledger": self.ledger.to_dict(),
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_RUN_WORKLOAD_FIELDS = {
    "workload_id",
    "workload_seed",
    "comparison_group_id",
    "pair_index",
    "sequence_index",
}


def _reference_context_from_records(
    ledger: EvidenceLedger, comparison_group_ids: Sequence[str]
) -> dict[str, str]:
    group_ids = set(comparison_group_ids)
    records = tuple(
        record
        for record in ledger.records
        if record.workload.get("comparison_group_id") in group_ids
    )
    if not records:
        raise ValueError("replay-noise ledger has no records for its control groups")
    if any(record.quality.grade != EvidenceGrade.A_FORMAL_PAIRED for record in records):
        raise ValueError("replay-noise control records must all be grade A")

    context_values = {
        "algorithm_context_sha256": {
            canonical_sha256(dict(record.algorithm)) for record in records
        },
        "workload_context_sha256": {
            canonical_sha256(
                {
                    name: value
                    for name, value in record.workload.items()
                    if name not in _RUN_WORKLOAD_FIELDS
                }
            )
            for record in records
        },
        "environment_context_sha256": {
            canonical_sha256(dict(record.environment)) for record in records
        },
        "baseline_settings_sha256": {
            canonical_sha256(
                {
                    name: value
                    for name, value in record.configuration.items()
                    if name != "configuration_id"
                }
            )
            for record in records
        },
    }
    inconsistent = sorted(
        name for name, values in context_values.items() if len(values) != 1
    )
    if inconsistent:
        raise ValueError(
            f"replay-noise control records disagree on context: {inconsistent}"
        )
    return {
        name: next(iter(values)) for name, values in context_values.items()
    }


def _plan_noise_context(plan: CalibrationPlan) -> dict[str, str]:
    semantic = plan.spec.semantic_contract
    workload = plan.spec.workload_contract
    return {
        "algorithm_context_sha256": canonical_sha256(
            {
                "algorithm_id": semantic.algorithm_id,
                "semantic_class": semantic.semantic_class,
                "graph_sha256": semantic.graph_sha256,
                **dict(semantic.invariants),
            }
        ),
        "workload_context_sha256": canonical_sha256(
            {
                "dataset_sha256": workload.dataset_sha256,
                "arrival_trace_sha256": workload.arrival_trace_sha256,
                **dict(workload.parameters),
            }
        ),
        "environment_context_sha256": canonical_sha256(
            plan.spec.environment_contract.to_dict()
        ),
        "baseline_settings_sha256": canonical_sha256(
            dict(plan.spec.baseline.settings)
        ),
    }


def load_replay_noise_reference(path: str | Path) -> ReplayNoiseReference:
    """Load a formal replay-control envelope with content-addressed provenance."""

    path = Path(path).expanduser().resolve()
    raw = require_object(
        json.loads(path.read_text(encoding="utf-8")),
        f"replay-noise assessment {path}",
    )
    if raw.get("formal_complete") is not True:
        raise ValueError("replay-noise assessment must be formally complete")
    envelope = raw.get("local_replay_noise_envelope", raw.get("replay_noise_envelope"))
    if (
        isinstance(envelope, bool)
        or not isinstance(envelope, (int, float))
        or not isfinite(float(envelope))
        or float(envelope) < 0
    ):
        raise ValueError("replay-noise assessment has no finite local envelope")
    effects_raw = raw.get("effects")
    if not isinstance(effects_raw, list):
        raise ValueError("replay-noise assessment effects must be an array")
    controls = [
        require_object(effect, "replay-control effect")
        for effect in effects_raw
        if isinstance(effect, Mapping)
        and effect.get("replay_control") is True
        and effect.get("formal_group") is True
        and effect.get("complete_pair_count") == effect.get("expected_pair_count")
    ]
    if not controls:
        raise ValueError("replay-noise assessment has no complete formal control")
    metrics = {str(effect.get("primary_metric")) for effect in controls}
    directions = {str(effect.get("direction")) for effect in controls}
    if len(metrics) != 1 or len(directions) != 1:
        raise ValueError("replay controls disagree on metric or direction")
    plan_sha256 = raw.get("plan_sha256")
    if (
        not isinstance(plan_sha256, str)
        or len(plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in plan_sha256)
    ):
        raise ValueError("replay-noise assessment has an invalid plan digest")
    group_ids = tuple(sorted(str(effect["comparison_group_id"]) for effect in controls))
    ledger_raw = raw.get("ledger")
    if not isinstance(ledger_raw, Mapping):
        raise ValueError("replay-noise assessment has no evidence ledger")
    context = _reference_context_from_records(
        EvidenceLedger.from_dict(ledger_raw), group_ids
    )
    return ReplayNoiseReference(
        assessment_sha256=_file_sha256(path),
        plan_sha256=plan_sha256,
        primary_metric=next(iter(metrics)),
        direction=next(iter(directions)),
        replay_noise_envelope=float(envelope),
        comparison_group_ids=group_ids,
        **context,
    )


def load_observations(path: str | Path) -> tuple[LoadedObservation, ...]:
    """Load one observation file or every JSON file in a directory."""

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        files = [path]
        root = path.parent
    else:
        files = sorted(item for item in path.rglob("*.json") if item.is_file())
        root = path
    observations: list[LoadedObservation] = []
    for observation_path in files:
        raw = json.loads(observation_path.read_text(encoding="utf-8"))
        observation = RunObservation.from_dict(
            require_object(raw, f"observation {observation_path}")
        )
        observations.append(
            LoadedObservation(
                observation=observation,
                path=observation_path.relative_to(root).as_posix(),
                sha256=_file_sha256(observation_path),
            )
        )
    return tuple(observations)


def _metric_value(metrics: Mapping[str, Any], path: str) -> float:
    current: Any = metrics
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(path)
        current = current[component]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise TypeError(path)
    value = float(current)
    if not isfinite(value):
        raise ValueError(path)
    return value


def _constraint_violations(
    metrics: Mapping[str, Any], constraints: Sequence[MetricConstraint]
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for constraint in constraints:
        try:
            observed = _metric_value(metrics, constraint.metric)
        except (KeyError, TypeError, ValueError):
            continue
        satisfied = (
            observed <= constraint.value
            if constraint.operator == "<="
            else observed >= constraint.value
        )
        if not satisfied:
            violations.append({**constraint.to_dict(), "observed": observed})
    return violations


def _groups(plan: CalibrationPlan) -> dict[str, list[PlannedRun]]:
    groups: dict[str, list[PlannedRun]] = {}
    for run in plan.runs:
        groups.setdefault(run.comparison_group_id, []).append(run)
    return groups


def _geometric_mean(values: Sequence[float]) -> float | None:
    if not values or any(value <= 0 or not isfinite(value) for value in values):
        return None
    return exp(sum(log(value) for value in values) / len(values))


def _comparison_effects(
    plan: CalibrationPlan,
    groups: Mapping[str, Sequence[PlannedRun]],
    loaded_by_id: Mapping[str, LoadedObservation],
    invalid_run_ids: set[str],
    formal_groups: set[str],
    referenced_replay_noise_envelope: float | None,
) -> tuple[tuple[ComparisonEffect, ...], float | None, float | None]:
    configurations = {plan.spec.baseline.configuration_id: plan.spec.baseline}
    configurations.update(
        {candidate.configuration_id: candidate for candidate in plan.spec.candidates}
    )
    provisional: list[ComparisonEffect] = []
    for group_id, runs in groups.items():
        candidate_ids = {
            run.configuration_id for run in runs if run.variant_role == "candidate"
        }
        if len(candidate_ids) != 1:
            continue
        candidate_id = next(iter(candidate_ids))
        replay_control = (
            configurations[candidate_id].settings == plan.spec.baseline.settings
        )
        pair_effects: list[PairedMetricEffect] = []
        pair_indexes = sorted({run.pair_index for run in runs})
        candidate_observations: list[RunObservation] = []
        for pair_index in pair_indexes:
            pair_runs = [run for run in runs if run.pair_index == pair_index]
            baselines = [run for run in pair_runs if run.variant_role == "baseline"]
            candidates = [run for run in pair_runs if run.variant_role == "candidate"]
            if len(baselines) != 1 or len(candidates) != 1:
                continue
            baseline_run, candidate_run = baselines[0], candidates[0]
            if (
                baseline_run.run_id in invalid_run_ids
                or candidate_run.run_id in invalid_run_ids
                or baseline_run.run_id not in loaded_by_id
                or candidate_run.run_id not in loaded_by_id
            ):
                continue
            baseline_observation = loaded_by_id[baseline_run.run_id].observation
            candidate_observation = loaded_by_id[candidate_run.run_id].observation
            try:
                baseline_value = _metric_value(
                    baseline_observation.metrics,
                    plan.spec.objective.primary_metric,
                )
                candidate_value = _metric_value(
                    candidate_observation.metrics,
                    plan.spec.objective.primary_metric,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if baseline_value <= 0 or candidate_value <= 0:
                continue
            ratio = candidate_value / baseline_value
            directional = (
                ratio - 1.0
                if plan.spec.objective.direction == "maximize"
                else baseline_value / candidate_value - 1.0
            )
            pair_effects.append(
                PairedMetricEffect(
                    pair_index=pair_index,
                    workload_seed=baseline_run.workload_seed,
                    baseline_run_id=baseline_run.run_id,
                    candidate_run_id=candidate_run.run_id,
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    candidate_over_baseline_ratio=ratio,
                    directional_relative_improvement=directional,
                )
            )
            candidate_observations.append(candidate_observation)
        constraints_satisfied = bool(candidate_observations) and all(
            not _constraint_violations(
                observation.metrics,
                plan.spec.objective.constraints,
            )
            for observation in candidate_observations
        )
        ratios = [effect.candidate_over_baseline_ratio for effect in pair_effects]
        improvements = [
            effect.directional_relative_improvement for effect in pair_effects
        ]
        provisional.append(
            ComparisonEffect(
                comparison_group_id=group_id,
                candidate_configuration_id=candidate_id,
                primary_metric=plan.spec.objective.primary_metric,
                direction=plan.spec.objective.direction,
                replay_control=replay_control,
                formal_group=group_id in formal_groups,
                quality_constraints_satisfied=constraints_satisfied,
                expected_pair_count=len(pair_indexes),
                complete_pair_count=len(pair_effects),
                pair_effects=tuple(pair_effects),
                candidate_over_baseline_geomean_ratio=_geometric_mean(ratios),
                median_directional_relative_improvement=(
                    median(improvements) if improvements else None
                ),
            )
        )
    control_variation = [
        abs(pair.directional_relative_improvement)
        for effect in provisional
        if effect.replay_control and effect.formal_group
        for pair in effect.pair_effects
    ]
    local_replay_noise_envelope = max(control_variation) if control_variation else None
    available_envelopes = [
        value
        for value in (
            local_replay_noise_envelope,
            referenced_replay_noise_envelope,
        )
        if value is not None
    ]
    replay_noise_envelope = max(available_envelopes) if available_envelopes else None
    effects = tuple(
        replace(
            effect,
            effect_exceeds_replay_noise=(
                effect.median_directional_relative_improvement
                > replay_noise_envelope
                if replay_noise_envelope is not None
                and not effect.replay_control
                and effect.formal_group
                and effect.quality_constraints_satisfied
                and effect.complete_pair_count == effect.expected_pair_count
                and effect.median_directional_relative_improvement is not None
                else None
            ),
            effect_outside_replay_noise=(
                abs(effect.median_directional_relative_improvement)
                > replay_noise_envelope
                if replay_noise_envelope is not None
                and not effect.replay_control
                and effect.formal_group
                and effect.quality_constraints_satisfied
                and effect.complete_pair_count == effect.expected_pair_count
                and effect.median_directional_relative_improvement is not None
                else None
            ),
        )
        for effect in provisional
    )
    return effects, local_replay_noise_envelope, replay_noise_envelope


def assess_calibration(
    plan: CalibrationPlan,
    loaded_observations: Sequence[LoadedObservation],
    replay_noise_reference: ReplayNoiseReference | None = None,
) -> CalibrationAssessment:
    """Assess protocol compliance and produce an evidence ledger."""

    plan_sha256 = canonical_sha256(plan.to_dict())
    if replay_noise_reference is not None:
        if (
            replay_noise_reference.primary_metric
            != plan.spec.objective.primary_metric
            or replay_noise_reference.direction != plan.spec.objective.direction
        ):
            raise ValueError(
                "replay-noise reference metric and direction must match the "
                "calibration objective"
            )
        expected_context = _plan_noise_context(plan)
        mismatches = sorted(
            name
            for name, expected in expected_context.items()
            if getattr(replay_noise_reference, name) != expected
        )
        if mismatches:
            raise ValueError(
                f"replay-noise reference context does not match calibration: {mismatches}"
            )
    expected_by_id = {run.run_id: run for run in plan.runs}
    groups = _groups(plan)
    issues: list[AssessmentIssue] = []
    issue_codes_by_group: dict[str, set[str]] = {group_id: set() for group_id in groups}
    invalid_run_ids: set[str] = set()

    loaded_by_id: dict[str, LoadedObservation] = {}
    for loaded in loaded_observations:
        run_id = loaded.observation.run_id
        if run_id in loaded_by_id:
            group_id = expected_by_id.get(run_id)
            comparison_group_id = None if group_id is None else group_id.comparison_group_id
            issues.append(
                AssessmentIssue(
                    "duplicate_observation",
                    f"multiple observations claim run id {run_id}",
                    comparison_group_id,
                    (run_id,),
                )
            )
            if comparison_group_id is not None:
                issue_codes_by_group[comparison_group_id].add("duplicate_observation")
            invalid_run_ids.add(run_id)
            continue
        loaded_by_id[run_id] = loaded

    unexpected_ids = sorted(set(loaded_by_id) - set(expected_by_id))
    if unexpected_ids:
        issues.append(
            AssessmentIssue(
                "unexpected_observation",
                "observations contain run ids absent from the calibration plan",
                run_ids=tuple(unexpected_ids),
            )
        )

    for group_id, expected_runs in groups.items():
        missing = [run.run_id for run in expected_runs if run.run_id not in loaded_by_id]
        if missing:
            issues.append(
                AssessmentIssue(
                    "missing_observation",
                    "comparison group is missing planned observations",
                    group_id,
                    tuple(missing),
                )
            )
            issue_codes_by_group[group_id].add("missing_observation")

    for run_id, loaded in loaded_by_id.items():
        run = expected_by_id.get(run_id)
        if run is None:
            continue
        group_id = run.comparison_group_id
        expected_digest = build_run_manifest(plan, run_id)["run_manifest_sha256"]
        if loaded.observation.run_manifest_sha256 != expected_digest:
            issues.append(
                AssessmentIssue(
                    "manifest_mismatch",
                    "observation does not bind to the expected immutable run manifest",
                    group_id,
                    (run_id,),
                )
            )
            issue_codes_by_group[group_id].add("manifest_mismatch")
            invalid_run_ids.add(run_id)
        if loaded.observation.status != "success":
            issues.append(
                AssessmentIssue(
                    "run_failed",
                    f"run ended with status {loaded.observation.status}",
                    group_id,
                    (run_id,),
                )
            )
            issue_codes_by_group[group_id].add("run_failed")
            invalid_run_ids.add(run_id)
            continue
        missing_metrics: list[str] = []
        for metric in plan.spec.required_metrics:
            try:
                _metric_value(loaded.observation.metrics, metric)
            except (KeyError, TypeError, ValueError):
                missing_metrics.append(metric)
        if missing_metrics:
            issues.append(
                AssessmentIssue(
                    "invalid_metrics",
                    f"required metrics are missing, non-numeric or non-finite: {missing_metrics}",
                    group_id,
                    (run_id,),
                )
            )
            issue_codes_by_group[group_id].add("invalid_metrics")
            invalid_run_ids.add(run_id)

    for group_id, expected_runs in groups.items():
        observed = [
            loaded_by_id[run.run_id]
            for run in expected_runs
            if run.run_id in loaded_by_id
        ]
        actual_order = [
            loaded.observation.run_id
            for loaded in sorted(observed, key=lambda item: item.observation.started_at_unix)
        ]
        expected_order = [run.run_id for run in expected_runs if run.run_id in loaded_by_id]
        if actual_order != expected_order:
            issues.append(
                AssessmentIssue(
                    "run_order_mismatch",
                    "actual start order differs from the planned protocol order",
                    group_id,
                    tuple(actual_order),
                )
            )
            issue_codes_by_group[group_id].add("run_order_mismatch")

    known_observations = [
        loaded for run_id, loaded in loaded_by_id.items() if run_id in expected_by_id
    ]
    chronological = sorted(
        known_observations, key=lambda item: item.observation.started_at_unix
    )
    for previous, current in zip(chronological, chronological[1:]):
        if current.observation.started_at_unix < previous.observation.finished_at_unix:
            affected_groups = {
                expected_by_id[previous.observation.run_id].comparison_group_id,
                expected_by_id[current.observation.run_id].comparison_group_id,
            }
            run_ids = (previous.observation.run_id, current.observation.run_id)
            for group_id in affected_groups:
                issues.append(
                    AssessmentIssue(
                        "overlapping_runs",
                        "calibration runs overlap in wall-clock time",
                        group_id,
                        run_ids,
                    )
                )
                issue_codes_by_group[group_id].add("overlapping_runs")

    if not plan.spec.strong_baseline:
        for group_id in groups:
            issues.append(
                AssessmentIssue(
                    "weak_baseline",
                    "calibration spec does not certify the baseline as the current strong baseline",
                    group_id,
                )
            )
            issue_codes_by_group[group_id].add("weak_baseline")

    formal_groups = {
        group_id
        for group_id, expected_runs in groups.items()
        if not issue_codes_by_group[group_id]
        and all(run.run_id in loaded_by_id for run in expected_runs)
    }
    effects, local_replay_noise_envelope, replay_noise_envelope = _comparison_effects(
        plan,
        groups,
        loaded_by_id,
        invalid_run_ids,
        formal_groups,
        None
        if replay_noise_reference is None
        else replay_noise_reference.replay_noise_envelope,
    )

    configurations = {plan.spec.baseline.configuration_id: plan.spec.baseline}
    configurations.update(
        {candidate.configuration_id: candidate for candidate in plan.spec.candidates}
    )
    records: list[EvidenceRecord] = []
    for run in plan.runs:
        loaded = loaded_by_id.get(run.run_id)
        if loaded is None:
            continue
        observation = loaded.observation
        group_issue_codes = sorted(issue_codes_by_group[run.comparison_group_id])
        if run.run_id in invalid_run_ids:
            quality = QualityAssessment(
                EvidenceGrade.X_EXCLUDED,
                EvidencePurpose.CONSTRAINT_ONLY,
                False,
                (
                    f"run failed formal validation with status {observation.status}",
                    f"group issues: {group_issue_codes}",
                ),
            )
        elif run.comparison_group_id in formal_groups:
            quality = QualityAssessment(
                EvidenceGrade.A_FORMAL_PAIRED,
                EvidencePurpose.SELECTOR_FIT_AND_CLAIM,
                True,
                (
                    f"complete {plan.spec.protocol.pattern} ordered comparison group",
                    "manifest binding, non-overlap, required metrics and strong baseline verified",
                ),
            )
        else:
            quality = QualityAssessment(
                EvidenceGrade.B_CONTROLLED_SINGLE,
                EvidencePurpose.CALIBRATION_ONLY,
                False,
                (
                    "successful controlled run whose comparison group did not pass the formal gate",
                    f"group issues: {group_issue_codes}",
                ),
            )

        violations = _constraint_violations(
            observation.metrics, plan.spec.objective.constraints
        )
        metrics = {
            **dict(observation.metrics),
            "run_status": observation.status,
            "constraint_violations": violations,
        }
        tags = [
            "calibration",
            run.variant_role,
            plan.spec.semantic_contract.semantic_class,
            "formal_group" if run.comparison_group_id in formal_groups else "nonformal_group",
        ]
        if violations:
            tags.append("constraint_violation")
        records.append(
            EvidenceRecord(
                record_id=f"{loaded.sha256[:16]}:{run.run_id}",
                campaign=plan.spec.campaign_id,
                variant=run.configuration_id,
                source=SourceArtifact(
                    loaded.path,
                    loaded.sha256,
                    "calibration_observation_v1",
                    run.run_id,
                ),
                workload={
                    "workload_id": plan.spec.workload_contract.workload_id,
                    "dataset_sha256": plan.spec.workload_contract.dataset_sha256,
                    "arrival_trace_sha256": (
                        plan.spec.workload_contract.arrival_trace_sha256
                    ),
                    "workload_seed": run.workload_seed,
                    "comparison_group_id": run.comparison_group_id,
                    "pair_index": run.pair_index,
                    "sequence_index": run.sequence_index,
                    **dict(plan.spec.workload_contract.parameters),
                },
                configuration={
                    "configuration_id": run.configuration_id,
                    **dict(configurations[run.configuration_id].settings),
                },
                algorithm={
                    "algorithm_id": plan.spec.semantic_contract.algorithm_id,
                    "semantic_class": plan.spec.semantic_contract.semantic_class,
                    "graph_sha256": plan.spec.semantic_contract.graph_sha256,
                    **dict(plan.spec.semantic_contract.invariants),
                },
                environment=plan.spec.environment_contract.to_dict(),
                metrics=metrics,
                quality=quality,
                tags=tuple(tags),
            )
        )

    return CalibrationAssessment(
        plan_sha256=plan_sha256,
        total_group_count=len(groups),
        formal_group_count=len(formal_groups),
        issues=tuple(issues),
        effects=effects,
        local_replay_noise_envelope=local_replay_noise_envelope,
        replay_noise_envelope=replay_noise_envelope,
        replay_noise_reference=replay_noise_reference,
        ledger=EvidenceLedger(records=tuple(records)),
    )
