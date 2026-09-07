"""Coverage-constrained search over vLLM graph capture bucket subsets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from inference_autopilot.calibration.models import canonical_sha256, require_id
from inference_autopilot.graph_corpus import GraphProfileCorpus
from inference_autopilot.graph_experiment import (
    GraphEnginePolicy,
    GraphExperimentPlan,
    GraphPolicy,
    build_graph_policy_experiment_plan,
    evaluate_graph_policy_coverage,
)
from inference_autopilot.vllm_graph_metrics import VLLMEngineGraphProfile


def _fraction(value: float, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{context} must be between zero and one")
    return float(value)


def _positive_int(value: int, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class GraphBucketConstraints:
    minimum_retained_graph_fraction: float
    maximum_remapped_graph_fraction: float
    maximum_added_padding_ratio: float

    def __post_init__(self) -> None:
        _fraction(
            self.minimum_retained_graph_fraction,
            "minimum_retained_graph_fraction",
        )
        _fraction(
            self.maximum_remapped_graph_fraction,
            "maximum_remapped_graph_fraction",
        )
        _fraction(self.maximum_added_padding_ratio, "maximum_added_padding_ratio")

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_retained_graph_fraction": self.minimum_retained_graph_fraction,
            "maximum_remapped_graph_fraction": self.maximum_remapped_graph_fraction,
            "maximum_added_padding_ratio": self.maximum_added_padding_ratio,
        }


@dataclass(frozen=True, slots=True)
class GraphBucketRoleSearch:
    engine_role: str
    configured_bucket_count: int
    search_strategy: str
    search_space_subset_count: int
    evaluated_transition_count: int
    retained_pareto_state_count: int
    selected_capture_sizes: tuple[int, ...]
    selected_train_worst_case: Mapping[str, float]
    bucket_count_frontier: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_role": self.engine_role,
            "configured_bucket_count": self.configured_bucket_count,
            "search_strategy": self.search_strategy,
            "search_space_subset_count": self.search_space_subset_count,
            "evaluated_transition_count": self.evaluated_transition_count,
            "retained_pareto_state_count": self.retained_pareto_state_count,
            "selected_capture_sizes": list(self.selected_capture_sizes),
            "selected_train_worst_case": dict(self.selected_train_worst_case),
            "bucket_count_frontier": [dict(point) for point in self.bucket_count_frontier],
        }


@dataclass(frozen=True, slots=True)
class GraphBucketPolicySearch:
    corpus_id: str
    corpus_sha256: str
    policy: GraphPolicy
    constraints: GraphBucketConstraints
    holdout_constraints: GraphBucketConstraints
    minimum_train_regimes: int
    minimum_holdout_regimes: int
    train_regime_count: int
    holdout_regime_count: int
    role_searches: tuple[GraphBucketRoleSearch, ...]
    coverage: tuple[Mapping[str, Any], ...]
    eligible: bool
    rejection_reasons: tuple[str, ...]
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "corpus_id": self.corpus_id,
            "corpus_sha256": self.corpus_sha256,
            "policy": self.policy.to_dict(),
            "constraints": self.constraints.to_dict(),
            "holdout_constraints": self.holdout_constraints.to_dict(),
            "minimum_train_regimes": self.minimum_train_regimes,
            "minimum_holdout_regimes": self.minimum_holdout_regimes,
            "train_regime_count": self.train_regime_count,
            "holdout_regime_count": self.holdout_regime_count,
            "role_searches": [search.to_dict() for search in self.role_searches],
            "coverage": [dict(item) for item in self.coverage],
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
        }

    def artifact_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {**payload, "graph_bucket_policy_search_sha256": canonical_sha256(payload)}


def _worst_case(metrics: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {
        "minimum_retained_graph_fraction": min(
            item["retained_graph_fraction"] for item in metrics
        ),
        "maximum_remapped_graph_fraction": max(
            item["remapped_graph_fraction"] for item in metrics
        ),
        "maximum_added_padding_ratio": max(
            item["added_padding_ratio"] for item in metrics
        ),
    }


def _is_feasible(
    worst: Mapping[str, float],
    constraints: GraphBucketConstraints,
) -> bool:
    epsilon = 1e-12
    return (
        worst["minimum_retained_graph_fraction"] + epsilon
        >= constraints.minimum_retained_graph_fraction
        and worst["maximum_remapped_graph_fraction"]
        <= constraints.maximum_remapped_graph_fraction + epsilon
        and worst["maximum_added_padding_ratio"]
        <= constraints.maximum_added_padding_ratio + epsilon
    )


@dataclass(frozen=True, slots=True)
class _PathLabel:
    resources: tuple[int, ...]
    capture_sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _EngineResourceModel:
    source_events: int
    source_tokens: int
    source_padding: int
    remap_limit: float
    candidate_padding_limit: float
    retained_events_by_bucket: tuple[int, ...]
    edge_resources: tuple[tuple[tuple[int, int], ...], ...]


def _build_resource_model(
    engine: VLLMEngineGraphProfile,
    configured: Sequence[int],
    constraints: GraphBucketConstraints,
) -> _EngineResourceModel:
    stats = tuple(stat for stat in engine.stats if stat.runtime_mode != "NONE")
    source_events = sum(stat.count for stat in stats)
    source_tokens = sum(stat.num_unpadded_tokens * stat.count for stat in stats)
    source_padding = sum(stat.num_paddings * stat.count for stat in stats)
    retained_events = tuple(
        sum(stat.count for stat in stats if stat.num_unpadded_tokens <= bucket)
        for bucket in configured
    )
    edges = []
    for previous_index in range(-1, len(configured)):
        previous = configured[previous_index] if previous_index >= 0 else 0
        row = []
        for current_index, current in enumerate(configured):
            if current_index <= previous_index:
                row.append((0, 0))
                continue
            covered = tuple(
                stat
                for stat in stats
                if previous < stat.num_unpadded_tokens <= current
            )
            row.append(
                (
                    sum(
                        stat.count
                        for stat in covered
                        if current != stat.num_padded_tokens
                    ),
                    sum(
                        (current - stat.num_unpadded_tokens) * stat.count
                        for stat in covered
                    ),
                )
            )
        edges.append(tuple(row))
    return _EngineResourceModel(
        source_events=source_events,
        source_tokens=source_tokens,
        source_padding=source_padding,
        remap_limit=constraints.maximum_remapped_graph_fraction * source_events,
        candidate_padding_limit=(
            source_padding + constraints.maximum_added_padding_ratio * source_tokens
        ),
        retained_events_by_bucket=retained_events,
        edge_resources=tuple(edges),
    )


def _within_resource_limits(
    resources: Sequence[int],
    models: Sequence[_EngineResourceModel],
) -> bool:
    epsilon = 1e-12
    return all(
        resources[2 * index] <= model.remap_limit + epsilon
        and resources[2 * index + 1]
        <= model.candidate_padding_limit + epsilon
        for index, model in enumerate(models)
    )


def _retention_meets_limit(
    end_index: int,
    models: Sequence[_EngineResourceModel],
    constraints: GraphBucketConstraints,
) -> bool:
    epsilon = 1e-12
    return all(
        (
            model.retained_events_by_bucket[end_index] / model.source_events
            if model.source_events
            else 1.0
        )
        + epsilon
        >= constraints.minimum_retained_graph_fraction
        for model in models
    )


def _label_worst_case(
    label: _PathLabel,
    end_index: int,
    models: Sequence[_EngineResourceModel],
) -> dict[str, float]:
    metrics = []
    for index, model in enumerate(models):
        metrics.append(
            {
                "retained_graph_fraction": (
                    model.retained_events_by_bucket[end_index] / model.source_events
                    if model.source_events
                    else 1.0
                ),
                "remapped_graph_fraction": (
                    label.resources[2 * index] / model.source_events
                    if model.source_events
                    else 0.0
                ),
                "added_padding_ratio": (
                    max(
                        0,
                        label.resources[2 * index + 1] - model.source_padding,
                    )
                    / model.source_tokens
                    if model.source_tokens
                    else 0.0
                ),
            }
        )
    return _worst_case(metrics)


def _insert_pareto_label(
    frontier: list[_PathLabel],
    candidate: _PathLabel,
) -> bool:
    """Insert a label unless an equal-or-better resource path already exists."""

    survivors = []
    for existing in frontier:
        if existing.resources == candidate.resources:
            if existing.capture_sizes <= candidate.capture_sizes:
                return False
            continue
        if all(
            left <= right
            for left, right in zip(existing.resources, candidate.resources, strict=True)
        ):
            return False
        if all(
            left <= right
            for left, right in zip(candidate.resources, existing.resources, strict=True)
        ):
            continue
        survivors.append(existing)
    survivors.append(candidate)
    frontier[:] = survivors
    return True


def _search_role(
    role: str,
    train_engines: Sequence[VLLMEngineGraphProfile],
    constraints: GraphBucketConstraints,
) -> GraphBucketRoleSearch:
    configured = train_engines[0].configured_capture_sizes
    if any(engine.configured_capture_sizes != configured for engine in train_engines):
        raise ValueError(f"configured capture sizes differ for engine role {role}")
    models = tuple(
        _build_resource_model(engine, configured, constraints)
        for engine in train_engines
    )
    best_by_count: dict[int, tuple[tuple[Any, ...], tuple[int, ...], dict[str, float]]] = {}
    evaluated_transitions = 0
    retained_states = 0
    current: dict[int, list[_PathLabel]] = {}
    for end_index, bucket in enumerate(configured):
        resources = tuple(
            value
            for model in models
            for value in model.edge_resources[0][end_index]
        )
        evaluated_transitions += 1
        if not _within_resource_limits(resources, models):
            continue
        current[end_index] = [_PathLabel(resources, (bucket,))]
        retained_states += 1

    for size_count in range(1, len(configured) + 1):
        for end_index, labels in current.items():
            if not _retention_meets_limit(end_index, models, constraints):
                continue
            for label in labels:
                worst = _label_worst_case(label, end_index, models)
                if not _is_feasible(worst, constraints):
                    continue
                rank = (
                    worst["maximum_remapped_graph_fraction"],
                    worst["maximum_added_padding_ratio"],
                    1.0 - worst["minimum_retained_graph_fraction"],
                    label.capture_sizes,
                )
                best = best_by_count.get(size_count)
                if best is None or rank < best[0]:
                    best_by_count[size_count] = (
                        rank,
                        label.capture_sizes,
                        worst,
                    )
        if size_count == len(configured):
            break
        following: dict[int, list[_PathLabel]] = {}
        for previous_index, labels in current.items():
            for end_index in range(previous_index + 1, len(configured)):
                edge = tuple(
                    value
                    for model in models
                    for value in model.edge_resources[previous_index + 1][end_index]
                )
                target = following.setdefault(end_index, [])
                for label in labels:
                    evaluated_transitions += 1
                    resources = tuple(
                        left + right
                        for left, right in zip(label.resources, edge, strict=True)
                    )
                    if not _within_resource_limits(resources, models):
                        continue
                    candidate = _PathLabel(
                        resources,
                        (*label.capture_sizes, configured[end_index]),
                    )
                    if _insert_pareto_label(target, candidate):
                        retained_states += 1
        current = {index: labels for index, labels in following.items() if labels}
        if not current:
            break
    if not best_by_count:
        raise ValueError(f"no feasible graph bucket subset for engine role {role}")
    selected_count = min(best_by_count)
    _rank, selected, selected_worst = best_by_count[selected_count]
    frontier = tuple(
        {
            "bucket_count": count,
            "capture_sizes": list(best_by_count[count][1]),
            **best_by_count[count][2],
        }
        for count in sorted(best_by_count)
    )
    return GraphBucketRoleSearch(
        engine_role=role,
        configured_bucket_count=len(configured),
        search_strategy="ordered_pareto_dynamic_programming",
        search_space_subset_count=(1 << len(configured)) - 1,
        evaluated_transition_count=evaluated_transitions,
        retained_pareto_state_count=retained_states,
        selected_capture_sizes=selected,
        selected_train_worst_case=selected_worst,
        bucket_count_frontier=frontier,
    )


def _coverage_meets_constraints(
    coverage: Mapping[str, Any],
    constraints: GraphBucketConstraints,
) -> bool:
    epsilon = 1e-12
    for engine in coverage["engines"]:
        graph_events = int(engine["source_graph_event_count"])
        remapped_fraction = (
            int(engine["remapped_graph_event_count"]) / graph_events
            if graph_events
            else 0.0
        )
        if (
            float(engine["retained_source_graph_fraction"]) + epsilon
            < constraints.minimum_retained_graph_fraction
            or remapped_fraction
            > constraints.maximum_remapped_graph_fraction + epsilon
            or float(engine["additional_padding_ratio"])
            > constraints.maximum_added_padding_ratio + epsilon
        ):
            return False
    return True


def search_coverage_constrained_graph_policy(
    corpus: GraphProfileCorpus,
    *,
    policy_id: str,
    constraints: GraphBucketConstraints,
    holdout_constraints: GraphBucketConstraints | None = None,
    candidate_engine_roles: Sequence[str] | None = None,
    minimum_train_regimes: int,
    minimum_holdout_regimes: int,
) -> GraphBucketPolicySearch:
    """Search under train constraints, then gate with independent holdout limits."""

    require_id(policy_id, "coverage-constrained graph policy_id")
    _positive_int(minimum_train_regimes, "minimum_train_regimes")
    _positive_int(minimum_holdout_regimes, "minimum_holdout_regimes")
    effective_holdout_constraints = holdout_constraints or constraints
    train_members = tuple(
        member for member in corpus.members if member.split == "train"
    )
    holdout_members = tuple(
        member for member in corpus.members if member.split == "holdout"
    )
    train_regime_count = len({member.regime_id for member in train_members})
    holdout_regime_count = len({member.regime_id for member in holdout_members})
    roles = sorted(engine.engine_role for engine in train_members[0].profile.engines)
    selected_roles = set(candidate_engine_roles or roles)
    unknown_roles = selected_roles.difference(roles)
    if unknown_roles:
        raise ValueError(
            "unknown candidate engine roles: " + ", ".join(sorted(unknown_roles))
        )
    if not selected_roles:
        raise ValueError("candidate_engine_roles cannot be empty")
    role_searches = []
    policies = []
    for role in roles:
        engines = [
            next(
                engine
                for engine in member.profile.engines
                if engine.engine_role == role
            )
            for member in train_members
        ]
        if role not in selected_roles:
            policies.append(
                GraphEnginePolicy(
                    engine_role=role,
                    graph_mode=engines[0].graph_mode,
                    capture_sizes=engines[0].configured_capture_sizes,
                )
            )
            continue
        role_search = _search_role(role, engines, constraints)
        role_searches.append(role_search)
        policies.append(
            GraphEnginePolicy(
                engine_role=role,
                graph_mode=engines[0].graph_mode,
                capture_sizes=role_search.selected_capture_sizes,
            )
        )
    policy = GraphPolicy(
        policy_id=policy_id,
        derivation="coverage_constrained",
        engines=tuple(policies),
    )
    default_sizes = {
        engine.engine_role: engine.configured_capture_sizes
        for engine in train_members[0].profile.engines
    }
    coverage_rows = []
    holdout_passes = True
    for member in sorted(
        corpus.members,
        key=lambda item: (item.split, item.regime_id, item.profile.profile_id),
    ):
        result = evaluate_graph_policy_coverage(policy, member.profile)
        artifact = result.artifact_dict()
        applied_constraints = (
            constraints
            if member.split == "train"
            else effective_holdout_constraints
        )
        passes = _coverage_meets_constraints(artifact, applied_constraints)
        coverage_rows.append(
            {
                "regime_id": member.regime_id,
                "split": member.split,
                "profile_id": member.profile.profile_id,
                "passes_constraints": passes,
                "coverage": artifact,
            }
        )
        if member.split == "holdout" and not passes:
            holdout_passes = False

    reasons = []
    if train_regime_count < minimum_train_regimes:
        reasons.append("insufficient_train_regimes")
    if holdout_regime_count < minimum_holdout_regimes:
        reasons.append("insufficient_holdout_regimes")
    if all(
        engine.capture_sizes == default_sizes[engine.engine_role]
        for engine in policy.engines
    ):
        reasons.append("no_policy_change")
    if not holdout_passes:
        reasons.append("holdout_constraint_violation")
    return GraphBucketPolicySearch(
        corpus_id=corpus.corpus_id,
        corpus_sha256=corpus.digest,
        policy=policy,
        constraints=constraints,
        holdout_constraints=effective_holdout_constraints,
        minimum_train_regimes=minimum_train_regimes,
        minimum_holdout_regimes=minimum_holdout_regimes,
        train_regime_count=train_regime_count,
        holdout_regime_count=holdout_regime_count,
        role_searches=tuple(role_searches),
        coverage=tuple(coverage_rows),
        eligible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def build_coverage_constrained_graph_experiment_plan(
    corpus: GraphProfileCorpus,
    search: GraphBucketPolicySearch,
    *,
    campaign_id: str,
    blocks: int,
    pair_seeds: Sequence[int],
    include_replay_control: bool = True,
) -> GraphExperimentPlan:
    if search.corpus_sha256 != corpus.digest:
        raise ValueError("graph bucket search belongs to a different corpus")
    if not search.eligible:
        raise ValueError(
            "graph bucket policy is not eligible: "
            + ", ".join(search.rejection_reasons)
        )
    return build_graph_policy_experiment_plan(
        corpus.members[0].profile,
        campaign_id=campaign_id,
        candidate_policies=(search.policy,),
        blocks=blocks,
        pair_seeds=pair_seeds,
        include_replay_control=include_replay_control,
        source_profile_sha256=corpus.digest,
    )
