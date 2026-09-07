"""Audit completed graph-policy ABBA runs and compute paired effects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import exp, isfinite, log
from pathlib import Path
import re
from statistics import median
from typing import Any

from inference_autopilot.calibration.models import canonical_sha256, require_object
from inference_autopilot.graph_experiment import (
    GraphExperimentPlan,
    GraphExperimentRun,
    GraphPolicy,
)
from inference_autopilot.vllm_graph_metrics import parse_vllm_graph_metrics


_CAPTURE_COST = re.compile(
    r"Graph capturing finished in ([0-9]+(?:\.[0-9]+)?) secs, "
    r"took ([0-9]+(?:\.[0-9]+)?) GiB"
)
_INIT_COST = re.compile(
    r"init engine \(profile, create kv cache, warmup model\) took "
    r"([0-9]+(?:\.[0-9]+)?) seconds"
)
_ENGINE_ROLES = ("base", "proposal")


@dataclass(frozen=True, slots=True)
class GraphAssessmentIssue:
    code: str
    message: str
    severity: str
    run_ids: tuple[str, ...] = ()
    pair_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "run_ids": list(self.run_ids),
            "pair_index": self.pair_index,
        }


@dataclass(frozen=True, slots=True)
class GraphRunEvidence:
    run_id: str
    policy_id: str
    workload_seed: int
    result_path: str
    result_sha256: str
    completed_qps: float
    mean_latency_seconds: float
    elapsed_seconds: float
    accuracy: float
    total_forward_token_slots: int
    request_count: int
    output_sha256: str
    semantic_answer_sha256: str
    provenance_complete: bool
    provenance: Mapping[str, str]
    graph_profiles: Mapping[str, Mapping[str, Any]]
    startup: Mapping[str, Mapping[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "policy_id": self.policy_id,
            "workload_seed": self.workload_seed,
            "result_path": self.result_path,
            "result_sha256": self.result_sha256,
            "completed_qps": self.completed_qps,
            "mean_latency_seconds": self.mean_latency_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "accuracy": self.accuracy,
            "total_forward_token_slots": self.total_forward_token_slots,
            "request_count": self.request_count,
            "output_sha256": self.output_sha256,
            "semantic_answer_sha256": self.semantic_answer_sha256,
            "provenance_complete": self.provenance_complete,
            "provenance": dict(sorted(self.provenance.items())),
            "graph_profiles": {
                role: dict(profile) for role, profile in sorted(self.graph_profiles.items())
            },
            "startup": {
                role: dict(costs) for role, costs in sorted(self.startup.items())
            },
        }


@dataclass(frozen=True, slots=True)
class GraphPairEvidence:
    pair_index: int
    workload_seed: int
    baseline_run_id: str
    candidate_run_id: str
    workload_config_exact: bool
    environment_exact: bool
    request_identity_exact: bool
    semantic_answers_exact: bool
    outputs_exact: bool
    compute_work_exact: bool
    steady_state_comparable: bool
    compared_request_count: int
    exact_output_count: int
    semantic_answer_match_count: int
    accuracy_candidate_minus_baseline: float
    ratios: Mapping[str, float]
    startup_deltas: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_index": self.pair_index,
            "workload_seed": self.workload_seed,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "workload_config_exact": self.workload_config_exact,
            "environment_exact": self.environment_exact,
            "request_identity_exact": self.request_identity_exact,
            "semantic_answers_exact": self.semantic_answers_exact,
            "outputs_exact": self.outputs_exact,
            "compute_work_exact": self.compute_work_exact,
            "steady_state_comparable": self.steady_state_comparable,
            "compared_request_count": self.compared_request_count,
            "exact_output_count": self.exact_output_count,
            "semantic_answer_match_count": self.semantic_answer_match_count,
            "accuracy_candidate_minus_baseline": (
                self.accuracy_candidate_minus_baseline
            ),
            "ratios": dict(sorted(self.ratios.items())),
            "startup_deltas": dict(sorted(self.startup_deltas.items())),
        }


@dataclass(frozen=True, slots=True)
class GraphExperimentAssessment:
    graph_experiment_plan_sha256: str
    comparison_id: str
    expected_run_count: int
    valid_run_count: int
    startup_comparable: bool
    steady_state_comparable: bool
    formal_claim_eligible: bool
    evidence_tier: str
    issues: tuple[GraphAssessmentIssue, ...]
    runs: tuple[GraphRunEvidence, ...]
    pairs: tuple[GraphPairEvidence, ...]
    summary: Mapping[str, Any]
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_experiment_plan_sha256": self.graph_experiment_plan_sha256,
            "comparison_id": self.comparison_id,
            "expected_run_count": self.expected_run_count,
            "valid_run_count": self.valid_run_count,
            "startup_comparable": self.startup_comparable,
            "steady_state_comparable": self.steady_state_comparable,
            "formal_claim_eligible": self.formal_claim_eligible,
            "evidence_tier": self.evidence_tier,
            "issues": [issue.to_dict() for issue in self.issues],
            "runs": [run.to_dict() for run in self.runs],
            "pairs": [pair.to_dict() for pair in self.pairs],
            "summary": dict(self.summary),
        }

    def artifact_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {
            **payload,
            "graph_experiment_assessment_sha256": canonical_sha256(payload),
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_run_provenance(
    path: Path,
    *,
    expected: GraphExperimentRun,
    result: Mapping[str, Any],
    result_sha256: str,
    runner_log_sha256: str,
) -> tuple[dict[str, str], bool]:
    if not path.is_file():
        return {}, False
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in fields:
            raise ValueError(f"duplicate run metadata field: {key}")
        fields[key] = value
    required = {
        "run_id",
        "host",
        "npu_id",
        "source_repo",
        "source_revision",
        "image",
        "requests",
        "workers",
        "config",
        "graph_plan",
        "graph_run_id",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"run metadata is missing fields: {missing}")
    if fields["run_id"] != expected.run_id or fields["graph_run_id"] != expected.run_id:
        raise ValueError(f"run metadata id mismatch for {expected.run_id}")
    if int(fields["requests"]) != result.get("requests") or int(
        fields["workers"]
    ) != result.get("workers"):
        raise ValueError(f"run metadata workload mismatch for {expected.run_id}")

    digest_fields = {
        "source_snapshot_sha256",
        "config_sha256",
        "dataset_sha256",
        "wrapper_sha256",
        "runner_log_sha256",
        "result_sha256",
    }
    present_digests = digest_fields & set(fields)
    if present_digests and present_digests != digest_fields:
        raise ValueError(
            "run metadata provenance hashes must be either complete or absent"
        )
    complete = present_digests == digest_fields
    if complete:
        for key in digest_fields:
            if re.fullmatch(r"[0-9a-f]{64}", fields[key]) is None:
                raise ValueError(f"invalid run metadata digest: {key}")
        if fields["runner_log_sha256"] != runner_log_sha256:
            raise ValueError(f"runner log SHA256 mismatch for {expected.run_id}")
        if fields["result_sha256"] != result_sha256:
            raise ValueError(f"result SHA256 mismatch for {expected.run_id}")
    environment_keys = (
        "host",
        "npu_id",
        "source_repo",
        "source_revision",
        "source_snapshot_sha256",
        "image",
        "config",
        "config_sha256",
        "dataset_sha256",
        "wrapper_sha256",
    )
    return ({key: fields[key] for key in environment_keys if key in fields}, complete)


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{context} must be finite")
    return parsed


def _positive_number(value: Any, context: str) -> float:
    parsed = _finite_number(value, context)
    if parsed <= 0:
        raise ValueError(f"{context} must be positive")
    return parsed


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _nested(raw: Mapping[str, Any], path: str) -> Any:
    current: Any = raw
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ValueError(f"missing result field: {path}")
        current = current[component]
    return current


def _parse_startup(
    log_text: str,
    result: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    captures = [(float(seconds), float(gib)) for seconds, gib in _CAPTURE_COST.findall(log_text)]
    inits = [float(seconds) for seconds in _INIT_COST.findall(log_text)]
    if len(captures) != len(_ENGINE_ROLES) or len(inits) != len(_ENGINE_ROLES):
        raise ValueError(
            "runner log must contain exactly two graph-capture and two engine-init costs"
        )
    startup = {
        role: {
            "graph_capture_seconds": captures[index][0],
            "graph_capture_gib": captures[index][1],
            "engine_init_seconds": inits[index],
        }
        for index, role in enumerate(_ENGINE_ROLES)
    }
    profile = result.get("inference_autopilot_profile")
    if profile is None:
        return startup
    profile = require_object(profile, "inference_autopilot_profile")
    engine_loads = require_object(
        profile.get("engine_load_seconds"),
        "inference_autopilot_profile engine_load_seconds",
    )
    if set(engine_loads) != set(_ENGINE_ROLES):
        raise ValueError("engine_load_seconds must contain exactly base and proposal")
    for role in _ENGINE_ROLES:
        startup[role]["wrapper_load_seconds"] = _positive_number(
            engine_loads[role], f"{role} wrapper engine load seconds"
        )
    return startup


def _request_identity(outputs: Sequence[Any]) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for index, item in enumerate(outputs):
        output = require_object(item, f"result output {index}")
        identities.append(
            {
                "request_index": output.get("request_index"),
                "problem_index": output.get("problem_index"),
                "gold_answer": output.get("gold_answer"),
            }
        )
    return identities


def _semantic_answers(outputs: Sequence[Any]) -> list[dict[str, Any]]:
    identities = _request_identity(outputs)
    answers: list[dict[str, Any]] = []
    for identity, item in zip(identities, outputs):
        output = require_object(item, "result output")
        answers.append({**identity, "numeric_answer": output.get("numeric_answer")})
    return answers


def _workload_config(result: Mapping[str, Any]) -> dict[str, Any]:
    algorithm = require_object(result.get("algorithm"), "result algorithm")
    return {
        "method": result.get("method"),
        "dtype": result.get("dtype"),
        "requests": result.get("requests"),
        "workers": result.get("workers"),
        "arrival_qps": result.get("arrival_qps"),
        "runtime": result.get("runtime"),
        "algorithm": {
            key: value for key, value in algorithm.items() if key != "totals"
        },
    }


def _geometric_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if any(value <= 0 or not isfinite(value) for value in values):
        return None
    return exp(sum(log(value) for value in values) / len(values))


def _policy_map(policy: GraphPolicy) -> dict[str, Any]:
    return {engine.engine_role: engine for engine in policy.engines}


def _load_run(
    plan: GraphExperimentPlan,
    plan_sha256: str,
    expected: GraphExperimentRun,
    policy: GraphPolicy,
    results_root: Path,
) -> tuple[GraphRunEvidence, dict[str, Any]]:
    run_root = results_root / expected.run_id
    result_path = run_root / "result.json"
    log_path = run_root / "runner.log"
    if not result_path.is_file():
        raise FileNotFoundError(f"missing result for {expected.run_id}: {result_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"missing runner log for {expected.run_id}: {log_path}")
    result = require_object(
        json.loads(result_path.read_text(encoding="utf-8")),
        f"graph experiment result {expected.run_id}",
    )
    binding = require_object(
        result.get("graph_experiment"), f"graph binding {expected.run_id}"
    )
    expected_binding = {
        "campaign_id": plan.campaign_id,
        "source_profile_sha256": plan.source_profile_sha256,
        "run": expected.to_dict(),
        "policy": policy.to_dict(),
        "policy_sha256": canonical_sha256(policy.to_dict()),
    }
    if binding != expected_binding:
        raise ValueError(f"result graph binding mismatch for {expected.run_id}")

    log_bytes = log_path.read_bytes()
    log_text = log_bytes.decode("utf-8")
    marker = (
        f"graph-plan-sha256={plan_sha256} run-id={expected.run_id} "
        f"policy-id={policy.policy_id} workload-seed={expected.workload_seed}"
    )
    if log_text.count(marker) != 1:
        raise ValueError(f"runner log plan binding mismatch for {expected.run_id}")
    profile = parse_vllm_graph_metrics(
        log_text,
        profile_id=f"{expected.run_id}.runtime",
        source_log_sha256=hashlib.sha256(log_bytes).hexdigest(),
    )
    expected_engines = _policy_map(policy)
    actual_engines = {engine.engine_role: engine for engine in profile.engines}
    if set(actual_engines) != set(expected_engines):
        raise ValueError(f"runner log engine roles mismatch for {expected.run_id}")
    for role, configured in expected_engines.items():
        actual = actual_engines[role]
        if (
            actual.graph_mode != configured.graph_mode
            or actual.configured_capture_sizes != configured.capture_sizes
        ):
            raise ValueError(
                f"runner log graph policy mismatch for {expected.run_id} role={role}"
            )

    outputs = result.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"result outputs must be a non-empty array for {expected.run_id}")
    request_count = _positive_int(result.get("requests"), "result requests")
    if len(outputs) != request_count:
        raise ValueError(f"result output count mismatch for {expected.run_id}")
    total_slots = _positive_int(
        _nested(result, "compute.total_forward_token_slots"),
        "total_forward_token_slots",
    )
    result_sha256 = _file_sha256(result_path)
    runner_log_sha256 = hashlib.sha256(log_bytes).hexdigest()
    provenance, provenance_complete = _parse_run_provenance(
        run_root / "run.meta",
        expected=expected,
        result=result,
        result_sha256=result_sha256,
        runner_log_sha256=runner_log_sha256,
    )
    evidence = GraphRunEvidence(
        run_id=expected.run_id,
        policy_id=expected.policy_id,
        workload_seed=expected.workload_seed,
        result_path=result_path.relative_to(results_root).as_posix(),
        result_sha256=result_sha256,
        completed_qps=_positive_number(result.get("completed_qps"), "completed_qps"),
        mean_latency_seconds=_positive_number(
            _nested(result, "latency_seconds.mean"), "mean latency"
        ),
        elapsed_seconds=_positive_number(result.get("elapsed_seconds"), "elapsed_seconds"),
        accuracy=_finite_number(result.get("accuracy"), "accuracy"),
        total_forward_token_slots=total_slots,
        request_count=request_count,
        output_sha256=canonical_sha256(outputs),
        semantic_answer_sha256=canonical_sha256(_semantic_answers(outputs)),
        provenance_complete=provenance_complete,
        provenance=provenance,
        graph_profiles={
            role: actual_engines[role].audit() for role in sorted(actual_engines)
        },
        startup=_parse_startup(log_text, result),
    )
    return evidence, result


def assess_graph_experiment(
    plan: GraphExperimentPlan,
    results_root: str | Path,
    *,
    comparison_id: str,
    minimum_requests_per_run: int = 32,
    minimum_blocks: int = 2,
) -> GraphExperimentAssessment:
    """Validate one comparison and report startup and steady-state evidence separately."""

    minimum_requests_per_run = _positive_int(
        minimum_requests_per_run, "minimum_requests_per_run"
    )
    minimum_blocks = _positive_int(minimum_blocks, "minimum_blocks")
    results_root = Path(results_root).expanduser().resolve()
    plan_sha256 = canonical_sha256(plan.to_dict())
    expected_runs = tuple(run for run in plan.runs if run.comparison_id == comparison_id)
    if not expected_runs:
        raise ValueError(f"unknown graph experiment comparison: {comparison_id}")
    policies = {policy.policy_id: policy for policy in plan.policies}
    issues: list[GraphAssessmentIssue] = []
    evidence_by_id: dict[str, GraphRunEvidence] = {}
    results_by_id: dict[str, dict[str, Any]] = {}
    for expected in expected_runs:
        try:
            evidence, result = _load_run(
                plan,
                plan_sha256,
                expected,
                policies[expected.policy_id],
                results_root,
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            issues.append(
                GraphAssessmentIssue(
                    "invalid_run_artifact",
                    str(error),
                    "error",
                    (expected.run_id,),
                    expected.pair_index,
                )
            )
            continue
        evidence_by_id[expected.run_id] = evidence
        results_by_id[expected.run_id] = result

    ordered_evidence = tuple(
        evidence_by_id[run.run_id]
        for run in expected_runs
        if run.run_id in evidence_by_id
    )
    pair_evidence: list[GraphPairEvidence] = []
    pair_indexes = sorted({run.pair_index for run in expected_runs})
    for pair_index in pair_indexes:
        pair_runs = [run for run in expected_runs if run.pair_index == pair_index]
        baselines = [run for run in pair_runs if run.variant_role == "baseline"]
        candidates = [run for run in pair_runs if run.variant_role == "candidate"]
        if len(baselines) != 1 or len(candidates) != 1:
            issues.append(
                GraphAssessmentIssue(
                    "invalid_pair_structure",
                    "each pair requires exactly one baseline and one candidate",
                    "error",
                    tuple(run.run_id for run in pair_runs),
                    pair_index,
                )
            )
            continue
        baseline_run, candidate_run = baselines[0], candidates[0]
        if baseline_run.workload_seed != candidate_run.workload_seed:
            issues.append(
                GraphAssessmentIssue(
                    "pair_seed_mismatch",
                    "paired baseline and candidate use different workload seeds",
                    "error",
                    (baseline_run.run_id, candidate_run.run_id),
                    pair_index,
                )
            )
            continue
        if (
            baseline_run.run_id not in evidence_by_id
            or candidate_run.run_id not in evidence_by_id
        ):
            continue
        baseline = evidence_by_id[baseline_run.run_id]
        candidate = evidence_by_id[candidate_run.run_id]
        baseline_result = results_by_id[baseline_run.run_id]
        candidate_result = results_by_id[candidate_run.run_id]
        baseline_outputs = baseline_result["outputs"]
        candidate_outputs = candidate_result["outputs"]
        workload_config_exact = _workload_config(
            baseline_result
        ) == _workload_config(candidate_result)
        environment_exact = baseline.provenance == candidate.provenance
        identity_exact = _request_identity(baseline_outputs) == _request_identity(
            candidate_outputs
        )
        semantic_exact = _semantic_answers(baseline_outputs) == _semantic_answers(
            candidate_outputs
        )
        outputs_exact = baseline_outputs == candidate_outputs
        compared_request_count = min(len(baseline_outputs), len(candidate_outputs))
        exact_output_count = sum(
            baseline_output == candidate_output
            for baseline_output, candidate_output in zip(
                baseline_outputs,
                candidate_outputs,
            )
        )
        baseline_semantics = _semantic_answers(baseline_outputs)
        candidate_semantics = _semantic_answers(candidate_outputs)
        semantic_answer_match_count = sum(
            baseline_answer == candidate_answer
            for baseline_answer, candidate_answer in zip(
                baseline_semantics,
                candidate_semantics,
            )
        )
        compute_exact = (
            baseline.total_forward_token_slots == candidate.total_forward_token_slots
        )
        steady_comparable = workload_config_exact and identity_exact and outputs_exact
        pair_run_ids = (baseline.run_id, candidate.run_id)
        if not workload_config_exact:
            issues.append(
                GraphAssessmentIssue(
                    "workload_config_mismatch",
                    "paired runs used different non-graph workload or algorithm settings",
                    "error",
                    pair_run_ids,
                    pair_index,
                )
            )
        if not environment_exact:
            issues.append(
                GraphAssessmentIssue(
                    "environment_mismatch",
                    "paired runs used different host, source, image, config or dataset",
                    "error",
                    pair_run_ids,
                    pair_index,
                )
            )
        if not identity_exact:
            issues.append(
                GraphAssessmentIssue(
                    "request_identity_mismatch",
                    "paired runs did not execute the same request identities",
                    "error",
                    pair_run_ids,
                    pair_index,
                )
            )
        if not outputs_exact:
            issues.append(
                GraphAssessmentIssue(
                    "stochastic_replay_mismatch",
                    "paired runs produced different token outputs despite sharing a seed",
                    "error",
                    pair_run_ids,
                    pair_index,
                )
            )
        if not compute_exact:
            issues.append(
                GraphAssessmentIssue(
                    "execution_shape_drift",
                    "paired runs executed different padded forward token slots",
                    "warning",
                    pair_run_ids,
                    pair_index,
                )
            )
        ratios = {
            "completed_qps_candidate_over_baseline": (
                candidate.completed_qps / baseline.completed_qps
            ),
            "mean_latency_candidate_over_baseline": (
                candidate.mean_latency_seconds / baseline.mean_latency_seconds
            ),
            "elapsed_candidate_over_baseline": (
                candidate.elapsed_seconds / baseline.elapsed_seconds
            ),
            "forward_token_slots_candidate_over_baseline": (
                candidate.total_forward_token_slots / baseline.total_forward_token_slots
            ),
        }
        baseline_capture = sum(
            costs["graph_capture_seconds"] for costs in baseline.startup.values()
        )
        candidate_capture = sum(
            costs["graph_capture_seconds"] for costs in candidate.startup.values()
        )
        baseline_init = sum(costs["engine_init_seconds"] for costs in baseline.startup.values())
        candidate_init = sum(costs["engine_init_seconds"] for costs in candidate.startup.values())
        baseline_wrapper_loads = [
            costs.get("wrapper_load_seconds") for costs in baseline.startup.values()
        ]
        candidate_wrapper_loads = [
            costs.get("wrapper_load_seconds") for costs in candidate.startup.values()
        ]
        wrapper_load_delta = None
        if all(value is not None for value in baseline_wrapper_loads) and all(
            value is not None for value in candidate_wrapper_loads
        ):
            wrapper_load_delta = sum(candidate_wrapper_loads) - sum(
                baseline_wrapper_loads
            )
        startup_deltas = {
            "graph_capture_seconds_candidate_minus_baseline": (
                candidate_capture - baseline_capture
            ),
            "engine_init_seconds_candidate_minus_baseline": (
                candidate_init - baseline_init
            ),
        }
        if wrapper_load_delta is not None:
            startup_deltas["wrapper_load_seconds_candidate_minus_baseline"] = (
                wrapper_load_delta
            )
        pair_evidence.append(
            GraphPairEvidence(
                pair_index=pair_index,
                workload_seed=baseline_run.workload_seed,
                baseline_run_id=baseline.run_id,
                candidate_run_id=candidate.run_id,
                workload_config_exact=workload_config_exact,
                environment_exact=environment_exact,
                request_identity_exact=identity_exact,
                semantic_answers_exact=semantic_exact,
                outputs_exact=outputs_exact,
                compute_work_exact=compute_exact,
                steady_state_comparable=steady_comparable,
                compared_request_count=compared_request_count,
                exact_output_count=exact_output_count,
                semantic_answer_match_count=semantic_answer_match_count,
                accuracy_candidate_minus_baseline=(
                    candidate.accuracy - baseline.accuracy
                ),
                ratios=ratios,
                startup_deltas=startup_deltas,
            )
        )

    request_counts = [run.request_count for run in ordered_evidence]
    if request_counts and min(request_counts) < minimum_requests_per_run:
        issues.append(
            GraphAssessmentIssue(
                "insufficient_requests",
                f"formal evidence requires at least {minimum_requests_per_run} requests per run",
                "warning",
                tuple(run.run_id for run in ordered_evidence),
            )
        )
    selected_blocks = len({run.block_index for run in expected_runs})
    if selected_blocks < minimum_blocks:
        issues.append(
            GraphAssessmentIssue(
                "insufficient_blocks",
                f"formal evidence requires at least {minimum_blocks} ABBA blocks",
                "warning",
                tuple(run.run_id for run in expected_runs),
            )
        )

    all_runs_valid = len(ordered_evidence) == len(expected_runs)
    complete_pairs = len(pair_evidence) == len(pair_indexes)
    startup_comparable = (
        all_runs_valid
        and complete_pairs
        and all(
            pair.workload_config_exact and pair.environment_exact
            for pair in pair_evidence
        )
    )
    steady_state_comparable = startup_comparable and all(
        pair.steady_state_comparable for pair in pair_evidence
    )
    formal_claim_eligible = (
        steady_state_comparable
        and selected_blocks >= minimum_blocks
        and bool(request_counts)
        and min(request_counts) >= minimum_requests_per_run
    )
    qps_ratios = [
        pair.ratios["completed_qps_candidate_over_baseline"] for pair in pair_evidence
    ]
    comparable_qps_ratios = [
        pair.ratios["completed_qps_candidate_over_baseline"]
        for pair in pair_evidence
        if pair.steady_state_comparable
    ]
    capture_deltas = [
        pair.startup_deltas["graph_capture_seconds_candidate_minus_baseline"]
        for pair in pair_evidence
    ]
    init_deltas = [
        pair.startup_deltas["engine_init_seconds_candidate_minus_baseline"]
        for pair in pair_evidence
    ]
    wrapper_load_deltas = [
        pair.startup_deltas["wrapper_load_seconds_candidate_minus_baseline"]
        for pair in pair_evidence
        if "wrapper_load_seconds_candidate_minus_baseline" in pair.startup_deltas
    ]
    summary = {
        "pair_count": len(pair_evidence),
        "steady_state_comparable_pair_count": sum(
            pair.steady_state_comparable for pair in pair_evidence
        ),
        "all_pairs_qps_geomean_ratio": _geometric_mean(qps_ratios),
        "comparable_pairs_qps_geomean_ratio": _geometric_mean(
            comparable_qps_ratios
        ),
        "median_graph_capture_seconds_delta": (
            median(capture_deltas) if capture_deltas else None
        ),
        "median_engine_init_seconds_delta": median(init_deltas) if init_deltas else None,
        "median_wrapper_load_seconds_delta": (
            median(wrapper_load_deltas) if wrapper_load_deltas else None
        ),
    }
    if formal_claim_eligible:
        evidence_tier = "formal_paired"
    elif startup_comparable or pair_evidence:
        evidence_tier = "diagnostic"
    else:
        evidence_tier = "invalid"
    return GraphExperimentAssessment(
        graph_experiment_plan_sha256=plan_sha256,
        comparison_id=comparison_id,
        expected_run_count=len(expected_runs),
        valid_run_count=len(ordered_evidence),
        startup_comparable=startup_comparable,
        steady_state_comparable=steady_state_comparable,
        formal_claim_eligible=formal_claim_eligible,
        evidence_tier=evidence_tier,
        issues=tuple(issues),
        runs=ordered_evidence,
        pairs=tuple(pair_evidence),
        summary=summary,
    )
