"""Build content-addressed ABBA experiments for vLLM graph policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from inference_autopilot.calibration.models import (
    CalibrationSpec,
    ConfigurationSpec,
    canonical_sha256,
    expect_keys,
    require_digest,
    require_id,
    require_object,
)
from inference_autopilot.vllm_graph_metrics import VLLMGraphMetricsProfile


_PATTERN = ("baseline", "candidate", "candidate", "baseline")


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class GraphEnginePolicy:
    engine_role: str
    graph_mode: str
    capture_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.engine_role not in {"base", "proposal"}:
            raise ValueError(f"unsupported graph policy engine role: {self.engine_role}")
        if not self.graph_mode or not self.graph_mode.replace("_", "").isalnum():
            raise ValueError(f"invalid graph policy mode: {self.graph_mode}")
        if tuple(sorted(set(self.capture_sizes))) != self.capture_sizes:
            raise ValueError("graph policy capture sizes must be unique and sorted")
        for size in self.capture_sizes:
            _positive_int(size, "graph policy capture size")
        if self.graph_mode == "NONE" and self.capture_sizes:
            raise ValueError("no-graph policy cannot specify capture sizes")
        if self.graph_mode != "NONE" and not self.capture_sizes:
            raise ValueError("enabled graph policy requires capture sizes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_role": self.engine_role,
            "graph_mode": self.graph_mode,
            "capture_sizes": list(self.capture_sizes),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphEnginePolicy":
        expect_keys(
            raw,
            {"engine_role", "graph_mode", "capture_sizes"},
            "graph engine policy",
        )
        sizes = raw["capture_sizes"]
        if not isinstance(sizes, list):
            raise ValueError("graph policy capture_sizes must be an array")
        return cls(
            engine_role=str(raw["engine_role"]),
            graph_mode=str(raw["graph_mode"]),
            capture_sizes=tuple(
                _positive_int(size, "graph policy capture size") for size in sizes
            ),
        )


@dataclass(frozen=True, slots=True)
class GraphPolicy:
    policy_id: str
    derivation: str
    engines: tuple[GraphEnginePolicy, ...]

    def __post_init__(self) -> None:
        require_id(self.policy_id, "graph policy_id")
        if self.derivation not in {
            "configured",
            "trace_preserving",
            "coverage_constrained",
            "no_graph",
        }:
            raise ValueError(f"unsupported graph policy derivation: {self.derivation}")
        roles = [engine.engine_role for engine in self.engines]
        if sorted(roles) != ["base", "proposal"]:
            raise ValueError("graph policy requires exactly base and proposal engines")
        if self.derivation == "no_graph" and any(
            engine.graph_mode != "NONE" for engine in self.engines
        ):
            raise ValueError("no_graph derivation requires NONE for every engine")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "derivation": self.derivation,
            "engines": [
                engine.to_dict()
                for engine in sorted(self.engines, key=lambda item: item.engine_role)
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphPolicy":
        expect_keys(raw, {"policy_id", "derivation", "engines"}, "graph policy")
        engines = raw["engines"]
        if not isinstance(engines, list):
            raise ValueError("graph policy engines must be an array")
        return cls(
            policy_id=str(raw["policy_id"]),
            derivation=str(raw["derivation"]),
            engines=tuple(
                GraphEnginePolicy.from_dict(require_object(item, "graph engine policy"))
                for item in engines
            ),
        )


@dataclass(frozen=True, slots=True)
class GraphExperimentRun:
    run_id: str
    comparison_id: str
    sequence_index: int
    group_sequence_index: int
    block_index: int
    pair_index: int
    variant_role: str
    policy_id: str
    workload_seed: int

    def __post_init__(self) -> None:
        require_id(self.run_id, "graph experiment run_id")
        require_id(self.comparison_id, "graph experiment comparison_id")
        require_id(self.policy_id, "graph experiment policy_id")
        for name in (
            "sequence_index",
            "group_sequence_index",
            "block_index",
            "pair_index",
            "workload_seed",
        ):
            _nonnegative_int(getattr(self, name), f"graph experiment {name}")
        if self.variant_role not in {"baseline", "candidate"}:
            raise ValueError("graph experiment variant_role must be baseline or candidate")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "comparison_id": self.comparison_id,
            "sequence_index": self.sequence_index,
            "group_sequence_index": self.group_sequence_index,
            "block_index": self.block_index,
            "pair_index": self.pair_index,
            "variant_role": self.variant_role,
            "policy_id": self.policy_id,
            "workload_seed": self.workload_seed,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphExperimentRun":
        expected = {
            "run_id",
            "comparison_id",
            "sequence_index",
            "group_sequence_index",
            "block_index",
            "pair_index",
            "variant_role",
            "policy_id",
            "workload_seed",
        }
        expect_keys(raw, expected, "graph experiment run")
        return cls(
            run_id=str(raw["run_id"]),
            comparison_id=str(raw["comparison_id"]),
            sequence_index=_nonnegative_int(raw["sequence_index"], "sequence_index"),
            group_sequence_index=_nonnegative_int(
                raw["group_sequence_index"], "group_sequence_index"
            ),
            block_index=_nonnegative_int(raw["block_index"], "block_index"),
            pair_index=_nonnegative_int(raw["pair_index"], "pair_index"),
            variant_role=str(raw["variant_role"]),
            policy_id=str(raw["policy_id"]),
            workload_seed=_nonnegative_int(raw["workload_seed"], "workload_seed"),
        )


@dataclass(frozen=True, slots=True)
class GraphExperimentPlan:
    campaign_id: str
    source_profile_sha256: str
    pattern: str
    blocks: int
    pair_seeds: tuple[int, ...]
    policies: tuple[GraphPolicy, ...]
    runs: tuple[GraphExperimentRun, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported graph experiment plan: {self.schema_version}")
        require_id(self.campaign_id, "graph experiment campaign_id")
        require_digest(self.source_profile_sha256, "graph experiment source profile SHA256")
        if self.pattern != "ABBA":
            raise ValueError("graph experiment pattern must be ABBA")
        _positive_int(self.blocks, "graph experiment blocks")
        if len(self.pair_seeds) != self.blocks * 2:
            raise ValueError("graph experiment requires two pair seeds per block")
        if len(set(self.pair_seeds)) != len(self.pair_seeds):
            raise ValueError("graph experiment pair seeds must be unique")
        for seed in self.pair_seeds:
            _nonnegative_int(seed, "graph experiment pair seed")
        policy_ids = [policy.policy_id for policy in self.policies]
        if not policy_ids or len(policy_ids) != len(set(policy_ids)):
            raise ValueError("graph experiment policy ids must be non-empty and unique")
        if "vllm-default" not in policy_ids:
            raise ValueError("graph experiment requires vllm-default policy")
        run_ids = [run.run_id for run in self.runs]
        if not run_ids or len(run_ids) != len(set(run_ids)):
            raise ValueError("graph experiment run ids must be non-empty and unique")
        if [run.sequence_index for run in self.runs] != list(range(len(self.runs))):
            raise ValueError("graph experiment sequence indexes must be contiguous")
        unknown_policies = sorted({run.policy_id for run in self.runs} - set(policy_ids))
        if unknown_policies:
            raise ValueError(f"graph experiment runs use unknown policies: {unknown_policies}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "source_profile_sha256": self.source_profile_sha256,
            "pattern": self.pattern,
            "blocks": self.blocks,
            "pair_seeds": list(self.pair_seeds),
            "policies": [policy.to_dict() for policy in self.policies],
            "runs": [run.to_dict() for run in self.runs],
        }

    def artifact_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {**payload, "graph_experiment_plan_sha256": canonical_sha256(payload)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphExperimentPlan":
        expected = {
            "schema_version",
            "campaign_id",
            "source_profile_sha256",
            "pattern",
            "blocks",
            "pair_seeds",
            "policies",
            "runs",
            "graph_experiment_plan_sha256",
        }
        expect_keys(raw, expected, "graph experiment plan")
        payload = dict(raw)
        digest = str(payload.pop("graph_experiment_plan_sha256"))
        require_digest(digest, "graph experiment plan SHA256")
        if canonical_sha256(payload) != digest:
            raise ValueError("graph experiment plan SHA256 does not match its content")
        seeds = payload["pair_seeds"]
        policies = payload["policies"]
        runs = payload["runs"]
        if not isinstance(seeds, list):
            raise ValueError("graph experiment pair_seeds must be an array")
        if not isinstance(policies, list) or not isinstance(runs, list):
            raise ValueError("graph experiment policies and runs must be arrays")
        return cls(
            schema_version=str(payload["schema_version"]),
            campaign_id=str(payload["campaign_id"]),
            source_profile_sha256=str(payload["source_profile_sha256"]),
            pattern=str(payload["pattern"]),
            blocks=_positive_int(payload["blocks"], "graph experiment blocks"),
            pair_seeds=tuple(
                _nonnegative_int(seed, "graph experiment pair seed") for seed in seeds
            ),
            policies=tuple(
                GraphPolicy.from_dict(require_object(item, "graph policy"))
                for item in policies
            ),
            runs=tuple(
                GraphExperimentRun.from_dict(
                    require_object(item, "graph experiment run")
                )
                for item in runs
            ),
        )

    def policy_for_run(self, run_id: str) -> tuple[GraphExperimentRun, GraphPolicy]:
        run = next((item for item in self.runs if item.run_id == run_id), None)
        if run is None:
            raise KeyError(f"unknown graph experiment run: {run_id}")
        policy = next(item for item in self.policies if item.policy_id == run.policy_id)
        return run, policy

    def audit(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "policy_count": len(self.policies),
            "run_count": len(self.runs),
            "comparisons": sorted({run.comparison_id for run in self.runs}),
            "policies": [policy.to_dict() for policy in self.policies],
        }


@dataclass(frozen=True, slots=True)
class GraphPolicyCoverage:
    profile_id: str
    profile_sha256: str
    policy_id: str
    policy_sha256: str
    engines: tuple[Mapping[str, Any], ...]
    mapping_exact: bool
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "profile_sha256": self.profile_sha256,
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "mapping_exact": self.mapping_exact,
            "engines": [dict(engine) for engine in self.engines],
        }

    def artifact_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {**payload, "graph_policy_coverage_sha256": canonical_sha256(payload)}


def build_graph_experiment_plan(
    profile: VLLMGraphMetricsProfile,
    *,
    campaign_id: str,
    blocks: int,
    pair_seeds: Sequence[int],
    include_replay_control: bool = False,
) -> GraphExperimentPlan:
    """Compare explicit framework defaults with trace pruning and no graph."""

    require_id(campaign_id, "graph experiment campaign_id")
    _positive_int(blocks, "graph experiment blocks")
    parsed_seeds = tuple(
        _nonnegative_int(seed, "graph experiment pair seed") for seed in pair_seeds
    )
    if len(parsed_seeds) != blocks * 2:
        raise ValueError("graph experiment requires two pair seeds per block")
    if len(set(parsed_seeds)) != len(parsed_seeds):
        raise ValueError("graph experiment pair seeds must be unique")
    default_engines = tuple(
        GraphEnginePolicy(
            engine_role=engine.engine_role,
            graph_mode=engine.graph_mode,
            capture_sizes=engine.configured_capture_sizes,
        )
        for engine in sorted(profile.engines, key=lambda item: item.engine_role)
    )
    pruned_engines = tuple(
        GraphEnginePolicy(
            engine_role=engine.engine_role,
            graph_mode=engine.graph_mode if engine.used_capture_sizes else "NONE",
            capture_sizes=engine.used_capture_sizes,
        )
        for engine in sorted(profile.engines, key=lambda item: item.engine_role)
    )
    policies = [GraphPolicy("vllm-default", "configured", default_engines)]
    if pruned_engines != default_engines:
        policies.append(
            GraphPolicy("trace-preserving-pruned", "trace_preserving", pruned_engines)
        )
    policies.append(
        GraphPolicy(
            "no-graph",
            "no_graph",
            tuple(
                GraphEnginePolicy(engine.engine_role, "NONE", ())
                for engine in default_engines
            ),
        )
    )

    runs: list[GraphExperimentRun] = []
    sequence_index = 0
    comparisons = [
        (policy.policy_id, policy.policy_id) for policy in policies[1:]
    ]
    if include_replay_control:
        comparisons.insert(0, ("replay-control", "vllm-default"))
    for comparison_name, candidate_policy_id in comparisons:
        comparison_id = f"{campaign_id}--{comparison_name}"
        group_index = 0
        for block_index in range(blocks):
            for position, variant_role in enumerate(_PATTERN):
                pair_index = block_index * 2 + position // 2
                policy_id = (
                    "vllm-default"
                    if variant_role == "baseline"
                    else candidate_policy_id
                )
                runs.append(
                    GraphExperimentRun(
                        run_id=f"{comparison_id}--{group_index:03d}-{variant_role}",
                        comparison_id=comparison_id,
                        sequence_index=sequence_index,
                        group_sequence_index=group_index,
                        block_index=block_index,
                        pair_index=pair_index,
                        variant_role=variant_role,
                        policy_id=policy_id,
                        workload_seed=parsed_seeds[pair_index],
                    )
                )
                sequence_index += 1
                group_index += 1

    return GraphExperimentPlan(
        campaign_id=campaign_id,
        source_profile_sha256=canonical_sha256(profile.to_dict()),
        pattern="ABBA",
        blocks=blocks,
        pair_seeds=parsed_seeds,
        policies=tuple(policies),
        runs=tuple(runs),
    )


def build_graph_policy_experiment_plan(
    profile: VLLMGraphMetricsProfile,
    *,
    campaign_id: str,
    candidate_policies: Sequence[GraphPolicy],
    blocks: int,
    pair_seeds: Sequence[int],
    include_replay_control: bool = True,
    source_profile_sha256: str | None = None,
) -> GraphExperimentPlan:
    """Build ABBA groups for externally planned graph-policy candidates."""

    require_id(campaign_id, "graph experiment campaign_id")
    _positive_int(blocks, "graph experiment blocks")
    parsed_seeds = tuple(
        _nonnegative_int(seed, "graph experiment pair seed") for seed in pair_seeds
    )
    if len(parsed_seeds) != blocks * 2:
        raise ValueError("graph experiment requires two pair seeds per block")
    if len(set(parsed_seeds)) != len(parsed_seeds):
        raise ValueError("graph experiment pair seeds must be unique")
    if not candidate_policies:
        raise ValueError("graph policy experiment requires candidate policies")

    default_engines = tuple(
        GraphEnginePolicy(
            engine_role=engine.engine_role,
            graph_mode=engine.graph_mode,
            capture_sizes=engine.configured_capture_sizes,
        )
        for engine in sorted(profile.engines, key=lambda item: item.engine_role)
    )
    default = GraphPolicy("vllm-default", "configured", default_engines)
    configured = {
        engine.engine_role: set(engine.capture_sizes) for engine in default.engines
    }
    configured_modes = {
        engine.engine_role: engine.graph_mode for engine in default.engines
    }
    policy_ids = []
    for policy in candidate_policies:
        if policy.policy_id == default.policy_id:
            raise ValueError("candidate policy cannot use the vllm-default id")
        if policy.engines == default.engines:
            raise ValueError(f"candidate policy {policy.policy_id} is a no-op")
        policy_ids.append(policy.policy_id)
        for engine in policy.engines:
            if engine.graph_mode not in {
                "NONE",
                configured_modes[engine.engine_role],
            }:
                raise ValueError(
                    f"candidate policy {policy.policy_id} changes the graph mode for "
                    f"{engine.engine_role}"
                )
            unknown = sorted(set(engine.capture_sizes) - configured[engine.engine_role])
            if unknown:
                raise ValueError(
                    f"candidate policy {policy.policy_id} uses unconfigured "
                    f"{engine.engine_role} buckets: {unknown}"
                )
    if len(policy_ids) != len(set(policy_ids)):
        raise ValueError("graph policy experiment candidate ids must be unique")

    runs: list[GraphExperimentRun] = []
    sequence_index = 0
    comparisons = [(policy.policy_id, policy.policy_id) for policy in candidate_policies]
    if include_replay_control:
        comparisons.insert(0, ("replay-control", default.policy_id))
    for comparison_name, candidate_policy_id in comparisons:
        comparison_id = f"{campaign_id}--{comparison_name}"
        group_index = 0
        for block_index in range(blocks):
            for position, variant_role in enumerate(_PATTERN):
                pair_index = block_index * 2 + position // 2
                policy_id = (
                    default.policy_id
                    if variant_role == "baseline"
                    else candidate_policy_id
                )
                runs.append(
                    GraphExperimentRun(
                        run_id=f"{comparison_id}--{group_index:03d}-{variant_role}",
                        comparison_id=comparison_id,
                        sequence_index=sequence_index,
                        group_sequence_index=group_index,
                        block_index=block_index,
                        pair_index=pair_index,
                        variant_role=variant_role,
                        policy_id=policy_id,
                        workload_seed=parsed_seeds[pair_index],
                    )
                )
                sequence_index += 1
                group_index += 1
    source_digest = source_profile_sha256 or canonical_sha256(profile.to_dict())
    require_digest(source_digest, "graph experiment source profile SHA256")
    return GraphExperimentPlan(
        campaign_id=campaign_id,
        source_profile_sha256=source_digest,
        pattern="ABBA",
        blocks=blocks,
        pair_seeds=parsed_seeds,
        policies=(default, *candidate_policies),
        runs=tuple(runs),
    )


def evaluate_graph_policy_coverage(
    policy: GraphPolicy,
    profile: VLLMGraphMetricsProfile,
) -> GraphPolicyCoverage:
    """Evaluate candidate bucket remapping on an independent shape profile."""

    policy_engines = {
        engine.engine_role: engine for engine in policy.engines
    }
    profile_engines = {
        engine.engine_role: engine for engine in profile.engines
    }
    if set(policy_engines) != set(profile_engines):
        raise ValueError("graph policy and coverage profile engine roles differ")
    audits: list[Mapping[str, Any]] = []
    for role in sorted(profile_engines):
        candidate = policy_engines[role]
        observed = profile_engines[role]
        if candidate.graph_mode not in {"NONE", observed.graph_mode}:
            raise ValueError(
                f"graph policy mode differs from coverage profile for role {role}"
            )
        total_events = sum(stat.count for stat in observed.stats)
        source_graph_events = 0
        candidate_graph_events = 0
        source_graph_token_units = 0
        remapped_events = 0
        dropped_events = 0
        source_padding = 0
        candidate_padding = 0
        remapped_shapes: list[dict[str, Any]] = []
        dropped_shapes: list[dict[str, Any]] = []
        for stat in observed.stats:
            if stat.runtime_mode == "NONE":
                continue
            source_graph_events += stat.count
            source_graph_token_units += stat.num_unpadded_tokens * stat.count
            source_padding += stat.num_paddings * stat.count
            target = next(
                (
                    size
                    for size in candidate.capture_sizes
                    if size >= stat.num_unpadded_tokens
                ),
                None,
            )
            if candidate.graph_mode == "NONE":
                target = None
            if target is None:
                dropped_events += stat.count
                dropped_shapes.append(
                    {
                        "num_unpadded_tokens": stat.num_unpadded_tokens,
                        "source_padded_tokens": stat.num_padded_tokens,
                        "count": stat.count,
                    }
                )
                continue
            candidate_graph_events += stat.count
            candidate_padding += (target - stat.num_unpadded_tokens) * stat.count
            if target != stat.num_padded_tokens:
                remapped_events += stat.count
                remapped_shapes.append(
                    {
                        "num_unpadded_tokens": stat.num_unpadded_tokens,
                        "source_padded_tokens": stat.num_padded_tokens,
                        "candidate_padded_tokens": target,
                        "count": stat.count,
                    }
                )
        audits.append(
            {
                "engine_role": role,
                "total_event_count": total_events,
                "source_graph_event_count": source_graph_events,
                "candidate_graph_event_count": candidate_graph_events,
                "source_graph_token_units": source_graph_token_units,
                "retained_source_graph_fraction": (
                    candidate_graph_events / source_graph_events
                    if source_graph_events
                    else 1.0
                ),
                "remapped_graph_event_count": remapped_events,
                "dropped_graph_event_count": dropped_events,
                "source_padding_units": source_padding,
                "candidate_padding_units": candidate_padding,
                "additional_padding_units": candidate_padding - source_padding,
                "additional_padding_ratio": (
                    max(0, candidate_padding - source_padding)
                    / source_graph_token_units
                    if source_graph_token_units
                    else 0.0
                ),
                "remapped_shapes": sorted(
                    remapped_shapes,
                    key=lambda item: (
                        item["num_unpadded_tokens"],
                        item["source_padded_tokens"],
                    ),
                ),
                "dropped_shapes": sorted(
                    dropped_shapes,
                    key=lambda item: (
                        item["num_unpadded_tokens"],
                        item["source_padded_tokens"],
                    ),
                ),
            }
        )
    mapping_exact = all(
        engine["remapped_graph_event_count"] == 0
        and engine["dropped_graph_event_count"] == 0
        for engine in audits
    )
    return GraphPolicyCoverage(
        profile_id=profile.profile_id,
        profile_sha256=canonical_sha256(profile.to_dict()),
        policy_id=policy.policy_id,
        policy_sha256=canonical_sha256(policy.to_dict()),
        engines=tuple(audits),
        mapping_exact=mapping_exact,
    )


def graph_policy_deployment_settings(policy: GraphPolicy) -> dict[str, Any]:
    """Translate a graph policy into chang adapter deployment settings."""

    settings: dict[str, Any] = {}
    for engine in sorted(policy.engines, key=lambda item: item.engine_role):
        settings[f"{engine.engine_role}_graph_mode"] = engine.graph_mode
        settings[f"{engine.engine_role}_graph_capture_sizes"] = list(
            engine.capture_sizes
        )
    return settings


def build_graph_policy_calibration_spec(
    template: CalibrationSpec,
    graph_plan: GraphExperimentPlan,
    *,
    campaign_id: str,
    candidate_policy_ids: Sequence[str] | None = None,
    include_replay_control: bool = False,
) -> CalibrationSpec:
    """Bind graph policies to a full workload and environment calibration contract."""

    require_id(campaign_id, "graph calibration campaign_id")
    policies = {policy.policy_id: policy for policy in graph_plan.policies}
    default = policies["vllm-default"]
    selected_ids = tuple(candidate_policy_ids or ())
    if not selected_ids:
        selected_ids = tuple(
            policy.policy_id
            for policy in graph_plan.policies
            if policy.policy_id != default.policy_id
        )
    if not selected_ids:
        raise ValueError("graph calibration requires at least one candidate policy")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("graph calibration candidate policy ids must be unique")
    if default.policy_id in selected_ids:
        raise ValueError("use include_replay_control for the default-policy control")
    unknown = sorted(set(selected_ids) - set(policies))
    if unknown:
        raise ValueError(f"unknown graph calibration policies: {unknown}")

    plan_sha256 = canonical_sha256(graph_plan.to_dict())
    base_settings = dict(template.baseline.settings)
    baseline = ConfigurationSpec(
        configuration_id=f"{template.baseline.configuration_id}-graph-default",
        description=(
            f"{template.baseline.description}; vLLM default graph policy from "
            f"graph plan {plan_sha256}"
        ),
        settings={**base_settings, **graph_policy_deployment_settings(default)},
    )
    candidates: list[ConfigurationSpec] = []
    if include_replay_control:
        candidates.append(
            ConfigurationSpec(
                configuration_id=(
                    f"{template.baseline.configuration_id}-graph-replay-control"
                ),
                description=(
                    "Identical default graph policy used to measure live-policy and "
                    f"harness variation; graph plan {plan_sha256}"
                ),
                settings=dict(baseline.settings),
            )
        )
    for policy_id in selected_ids:
        policy = policies[policy_id]
        candidates.append(
            ConfigurationSpec(
                configuration_id=(
                    f"{template.baseline.configuration_id}-graph-{policy.policy_id}"
                ),
                description=(
                    f"Graph policy {policy.policy_id} ({policy.derivation}) derived "
                    f"from graph plan {plan_sha256}"
                ),
                settings={
                    **base_settings,
                    **graph_policy_deployment_settings(policy),
                },
            )
        )

    return CalibrationSpec(
        campaign_id=campaign_id,
        strong_baseline=template.strong_baseline,
        protocol=template.protocol,
        semantic_contract=template.semantic_contract,
        workload_contract=template.workload_contract,
        environment_contract=template.environment_contract,
        objective=template.objective,
        required_metrics=template.required_metrics,
        baseline=baseline,
        candidates=tuple(candidates),
    )


def apply_graph_policy_to_config(
    config: dict[str, Any],
    policy: GraphPolicy,
    *,
    workload_seed: int,
) -> None:
    """Inject one role-specific policy into chang's parsed TOML configuration."""

    _nonnegative_int(workload_seed, "graph experiment workload seed")
    vllm = config.setdefault("vllm", {})
    if not isinstance(vllm, dict):
        raise ValueError("vllm config must be a table")
    common_engine = vllm.setdefault("engine_kwargs", {})
    if not isinstance(common_engine, dict):
        raise ValueError("vllm.engine_kwargs must be a table")
    common_engine.pop("compilation_config", None)
    common_engine["cudagraph_metrics"] = True
    for engine in policy.engines:
        role_config = vllm.setdefault(engine.engine_role, {})
        if not isinstance(role_config, dict):
            raise ValueError(f"vllm.{engine.engine_role} must be a table")
        role_engine = role_config.setdefault("engine_kwargs", {})
        if not isinstance(role_engine, dict):
            raise ValueError(f"vllm.{engine.engine_role}.engine_kwargs must be a table")
        compilation = {"cudagraph_mode": engine.graph_mode}
        if engine.capture_sizes:
            compilation["cudagraph_capture_sizes"] = list(engine.capture_sizes)
        role_engine["compilation_config"] = compilation
    run = config.setdefault("run", {})
    if not isinstance(run, dict):
        raise ValueError("run config must be a table")
    run["seed"] = workload_seed
