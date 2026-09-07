"""Command-line entry point for graph generation and evidence ingestion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from inference_autopilot.adapters import build_conditional_is_small_proposal_graph
from inference_autopilot.calibration import (
    CalibrationPlan,
    CalibrationSpec,
    assess_calibration,
    build_plan,
    build_run_manifest,
    load_observations,
    load_replay_noise_reference,
)
from inference_autopilot.calibration.models import RunObservation, require_object
from inference_autopilot.candidate_planning import (
    CandidateDesignSpec,
    CandidatePlan,
    build_candidate_plan,
    calibration_spec_from_candidate_plan,
    configurations_from_plan,
)
from inference_autopilot.capacity_graph_repair import (
    CapacityGraphRepairPlan,
    calibration_spec_from_capacity_graph_repair,
    plan_capacity_graph_repair,
)
from inference_autopilot.evidence import EvidenceLedger, merge_evidence_ledgers
from inference_autopilot.execution_readiness import (
    ExecutionReadinessAssessment,
    ExecutionReadinessSpec,
    assess_execution_readiness,
)
from inference_autopilot.features import (
    SelectorFeatureTable,
    features_from_ledger,
    features_from_run,
)
from inference_autopilot.graph_capture import (
    CapturePlan,
    CapturePlanningSpec,
    plan_graph_capture,
)
from inference_autopilot.graph_assessment import assess_graph_experiment
from inference_autopilot.graph_bucket_search import (
    GraphBucketConstraints,
    build_coverage_constrained_graph_experiment_plan,
    search_coverage_constrained_graph_policy,
)
from inference_autopilot.graph_corpus import (
    GraphProfileCorpus,
    build_graph_profile_corpus,
    build_promoted_graph_experiment_plan,
    promote_trace_preserving_graph_policy,
)
from inference_autopilot.graph_experiment import (
    GraphExperimentPlan,
    build_graph_experiment_plan,
    build_graph_policy_calibration_spec,
    evaluate_graph_policy_coverage,
)
from inference_autopilot.graph_trace import (
    GraphWorkloadTrace,
    trace_from_chang_result,
)
from inference_autopilot.importers import import_results
from inference_autopilot.ir import InferenceGraph
from inference_autopilot.mechanism_probe import (
    MechanismProbeAssessment,
    MechanismProbeAssessmentSpec,
    MechanismProbePlan,
    MechanismProbeSpec,
    assess_mechanism_probe,
    calibration_spec_from_mechanism_probe_plan,
    plan_mechanism_probes,
)
from inference_autopilot.npu_cost_model import (
    NPUCalibrationCorpus,
    NPUStageCostModel,
    fit_npu_stage_cost_model,
    predict_serial_graph_cost,
    stage_queries_from_dict,
)
from inference_autopilot.ordered_feasibility import (
    OrderedFeasibilityPlan,
    OrderedFeasibilitySpec,
    plan_ordered_feasibility,
)
from inference_autopilot.policy_selection import (
    PolicyBundle,
    PolicySelectionSpec,
    select_policy,
)
from inference_autopilot.policy_acquisition import (
    PolicyAcquisitionSpec,
    PolicyExperimentAssessment,
    PolicyExperimentAssessmentSpec,
    PolicyExperimentPlan,
    assess_policy_experiment,
    calibration_spec_from_policy_experiment_plan,
    plan_policy_experiments,
)
from inference_autopilot.policy_validation import (
    PolicyHoldoutAssessment,
    PolicyHoldoutSpec,
    assess_policy_holdout,
)
from inference_autopilot.stage_wavefront import plan_stage_wavefront
from inference_autopilot.policy_transfer import (
    PolicyTransferPlan,
    PolicyTransferSpec,
    calibration_spec_from_policy_transfer_plan,
    plan_policy_transfer,
)
from inference_autopilot.transfer_validation import (
    PolicyTransferAssessment,
    PolicyTransferAssessmentSpec,
    assess_policy_transfer,
)
from inference_autopilot.guarded_routing import (
    RuntimePolicyDecision,
    RuntimePolicyPool,
    RuntimePolicyPoolSpec,
    RuntimeRoutingRequest,
    compile_runtime_policy_pool,
    route_runtime_policy,
)
from inference_autopilot.harness_cost import (
    HarnessCostAssessment,
    assess_harness_cost,
)
from inference_autopilot.lifecycle_planning import (
    EngineLifecyclePlan,
    EngineLifecyclePlanningSpec,
    plan_engine_lifecycle,
)
from inference_autopilot.lifecycle_execution import (
    LifecycleExecutionReceipt,
    audit_lifecycle_execution,
)
from inference_autopilot.runners import (
    build_chang_run_bundle,
    observation_from_chang_result,
)
from inference_autopilot.runners.chang import load_manifest
from inference_autopilot.runtime_closure import (
    RuntimeClosureAssessment,
    attest_runtime_closure,
    enrich_features_with_runtime_closure,
)
from inference_autopilot.scoring import (
    ScoreWorkload,
    build_score_reduction_plan,
    requirement_for_reward,
)
from inference_autopilot.sequential_effect import (
    SequentialEffectAssessment,
    SequentialEffectSpec,
    assess_sequential_effect_files,
    calibration_spec_from_sequential_effect,
)
from inference_autopilot.sequential_design import (
    SequentialDesignStudy,
    SequentialDesignStudySpec,
    build_sequential_design_study,
)
from inference_autopilot.search_space import (
    CompiledSearchSpace,
    DeploymentSearchSpace,
    RuntimeCapabilityProfile,
    compile_search_space,
    configuration_from_candidate,
)
from inference_autopilot.vllm_graph_metrics import (
    VLLMGraphMetricsProfile,
    merge_vllm_graph_metrics,
    parse_vllm_graph_metrics,
)


def _write_json(payload: dict[str, Any], output: Path | None) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if output is None:
        sys.stdout.write(serialized)
        return
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)


def _load_ledger(path: Path) -> EvidenceLedger:
    raw = _load_object(path, "evidence ledger or calibration assessment")
    embedded = raw.get("ledger")
    if embedded is not None:
        raw = require_object(embedded, "embedded evidence ledger")
    return EvidenceLedger.from_dict(raw)


def _load_feature_table(path: Path) -> SelectorFeatureTable:
    return SelectorFeatureTable.from_dict(_load_object(path, "selector feature table"))


def _load_compiled_space(path: Path) -> CompiledSearchSpace:
    return CompiledSearchSpace.from_dict(_load_object(path, "compiled search space"))


def _load_candidate_plan(path: Path) -> CandidatePlan:
    return CandidatePlan.from_dict(_load_object(path, "candidate plan"))


def _load_capture_plan(path: Path) -> CapturePlan:
    return CapturePlan.from_dict(_load_object(path, "capture plan"))


def _load_ordered_feasibility_plan(path: Path) -> OrderedFeasibilityPlan:
    return OrderedFeasibilityPlan.from_dict(
        _load_object(path, "ordered feasibility plan")
    )


def _load_object(path: Path, context: str) -> dict[str, Any]:
    raw = json.loads(path.expanduser().read_text(encoding="utf-8"))
    return require_object(raw, context)


def _load_regime_profiles(
    assignments: Sequence[str],
) -> list[tuple[str, VLLMGraphMetricsProfile]]:
    profiles = []
    for assignment in assignments:
        regime_id, separator, raw_path = assignment.partition("=")
        if not separator or not regime_id or not raw_path:
            raise ValueError("graph profile assignment must be REGIME_ID=PROFILE.json")
        profile = VLLMGraphMetricsProfile.from_dict(
            _load_object(Path(raw_path), f"vLLM graph profile for {regime_id}")
        )
        profiles.append((regime_id, profile))
    return profiles


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inference-autopilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    graph = subparsers.add_parser("graph", help="emit an Inference Graph IR document")
    graph.add_argument("--candidate-count", type=int, default=4)
    graph.add_argument("--rollout-count", type=int, default=4)
    graph.add_argument("--block-size", type=int, default=16)
    graph.add_argument("--total-length", type=int, default=128)
    graph.add_argument(
        "--disable-importance-correction",
        action="store_true",
        help="emit the explicitly biased small-proposal ablation without target scoring",
    )
    graph.add_argument("--output", type=Path)

    ingest = subparsers.add_parser(
        "import-results", help="normalize legacy inference-scaling JSON"
    )
    ingest.add_argument("source", type=Path)
    ingest.add_argument("--output", type=Path)
    ingest.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero status when any source artifact is rejected",
    )

    audit = subparsers.add_parser(
        "audit", help="summarize a normalized evidence ledger"
    )
    audit.add_argument("ledger", type=Path)

    merge_ledgers = subparsers.add_parser(
        "merge-ledgers",
        help="merge compatible ledgers or assessments with conflict checks",
    )
    merge_ledgers.add_argument("ledgers", type=Path, nargs="+")
    merge_ledgers.add_argument("--output", type=Path)

    plan = subparsers.add_parser(
        "plan-calibration", help="expand a calibration spec into ABBA/BAAB runs"
    )
    plan.add_argument("spec", type=Path)
    plan.add_argument("--output", type=Path)

    manifest = subparsers.add_parser(
        "run-manifest", help="emit the immutable manifest for one planned run"
    )
    manifest.add_argument("plan", type=Path)
    manifest.add_argument("run_id")
    manifest.add_argument("--output", type=Path)

    assess = subparsers.add_parser(
        "assess-calibration", help="grade completed observations against a plan"
    )
    assess.add_argument("plan", type=Path)
    assess.add_argument("observations", type=Path)
    assess.add_argument(
        "--replay-noise-assessment",
        type=Path,
        help="formal replay-control assessment used as the effect-size floor",
    )
    assess.add_argument("--output", type=Path)

    harness_cost = subparsers.add_parser(
        "assess-harness-cost",
        help="extract startup, compilation, graph-capture, and cache reuse costs",
    )
    harness_cost.add_argument("plan", type=Path)
    harness_cost.add_argument("campaign_root", type=Path)
    harness_cost.add_argument("--output", type=Path)

    harness_cost_audit = subparsers.add_parser(
        "audit-harness-cost",
        help="verify and summarize content-addressed harness cost evidence",
    )
    harness_cost_audit.add_argument("assessment", type=Path)

    runtime_closure = subparsers.add_parser(
        "attest-runtime-closure",
        help="bind requested settings to the configuration resolved by vLLM",
    )
    runtime_closure.add_argument("plan", type=Path)
    runtime_closure.add_argument("campaign_root", type=Path)
    runtime_closure.add_argument("--output", type=Path)

    runtime_closure_audit = subparsers.add_parser(
        "audit-runtime-closure",
        help="verify and summarize content-addressed runtime closure evidence",
    )
    runtime_closure_audit.add_argument("assessment", type=Path)

    runtime_closure_features = subparsers.add_parser(
        "enrich-runtime-features",
        help="join attested runtime defaults into a selector feature table",
    )
    runtime_closure_features.add_argument("features", type=Path)
    runtime_closure_features.add_argument("assessment", type=Path)
    runtime_closure_features.add_argument("--output", type=Path)

    lifecycle_plan = subparsers.add_parser(
        "plan-engine-lifecycle",
        help="plan state-isolated engine reuse from measured startup evidence",
    )
    lifecycle_plan.add_argument("spec", type=Path)
    lifecycle_plan.add_argument("plan", type=Path)
    lifecycle_plan.add_argument("harness_cost_assessment", type=Path)
    lifecycle_plan.add_argument("--output", type=Path)

    lifecycle_audit = subparsers.add_parser(
        "audit-engine-lifecycle",
        help="verify and summarize a content-addressed engine lifecycle plan",
    )
    lifecycle_audit.add_argument("lifecycle_plan", type=Path)

    lifecycle_execution_audit = subparsers.add_parser(
        "audit-lifecycle-execution",
        help="verify a validation-only execution receipt against its lifecycle plan",
    )
    lifecycle_execution_audit.add_argument("lifecycle_plan", type=Path)
    lifecycle_execution_audit.add_argument("execution_receipt", type=Path)

    prepare = subparsers.add_parser(
        "prepare-run", help="prepare a dry-run bundle for chang's pressure runner"
    )
    prepare.add_argument("plan", type=Path)
    prepare.add_argument("run_id")
    prepare.add_argument("--source-repo", type=Path, required=True)
    prepare.add_argument("--source-config", type=Path, required=True)
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--python", default="python3")
    prepare.add_argument(
        "--require-formal",
        action="store_true",
        help="return non-zero when static checks cannot certify a formal run",
    )

    observe = subparsers.add_parser(
        "observe-run", help="convert a manifest-bound chang result into an observation"
    )
    observe.add_argument("manifest", type=Path)
    observe.add_argument("native_result", type=Path)
    observe.add_argument("--output", type=Path, required=True)

    ledger_features = subparsers.add_parser(
        "features-ledger",
        help="extract selector features from a graded evidence ledger",
    )
    ledger_features.add_argument("ledger", type=Path)
    ledger_features.add_argument("--output", type=Path)

    run_features = subparsers.add_parser(
        "features-run",
        help="extract pending features from a manifest-bound observation",
    )
    run_features.add_argument("manifest", type=Path)
    run_features.add_argument("observation", type=Path)
    run_features.add_argument("--output", type=Path)

    feature_audit = subparsers.add_parser(
        "audit-features", help="verify and summarize a selector feature table"
    )
    feature_audit.add_argument("table", type=Path)

    npu_cost_fit = subparsers.add_parser(
        "fit-npu-cost-model",
        help="fit uncertainty-aware affine stage models inside measured NPU buckets",
    )
    npu_cost_fit.add_argument("calibration", type=Path)
    npu_cost_fit.add_argument("--output", type=Path)

    npu_cost_predict = subparsers.add_parser(
        "predict-npu-graph-cost",
        help="compose calibrated NPU stage predictions as a serial upper bound",
    )
    npu_cost_predict.add_argument("model", type=Path)
    npu_cost_predict.add_argument("query", type=Path)
    npu_cost_predict.add_argument("--output", type=Path)

    compile_space = subparsers.add_parser(
        "compile-space", help="compile a constrained deployment search space"
    )
    compile_space.add_argument("space", type=Path)
    compile_space.add_argument(
        "--graph", type=Path, help="validate knob stage bindings against this graph"
    )
    compile_space.add_argument(
        "--capabilities",
        type=Path,
        help="reject knob values unsupported by this runtime capability profile",
    )
    compile_space.add_argument("--output", type=Path)
    compile_space.add_argument("--max-cartesian-product", type=int, default=100_000)
    compile_space.add_argument(
        "--allow-algorithm-changes",
        action="store_true",
        help="compile quality-sensitive knobs into separate semantic cohorts",
    )

    score_plan = subparsers.add_parser(
        "plan-score-reduction",
        help="enumerate memory-feasible exact trajectory scoring reductions",
    )
    score_plan.add_argument(
        "--reward",
        required=True,
        choices=(
            "self_consistency",
            "frozen_consensus",
            "exact",
            "consilience",
            "self_certainty",
            "entropy",
        ),
    )
    score_plan.add_argument("--importance-correction", action="store_true")
    score_plan.add_argument("--consilience-top-k", type=int, default=5)
    score_plan.add_argument("--positions", type=int, required=True)
    score_plan.add_argument("--vocab-size", type=int, required=True)
    score_plan.add_argument("--memory-budget-gib", type=float, required=True)
    score_plan.add_argument("--logits-element-bytes", type=int, default=2)
    score_plan.add_argument("--accumulator-bytes", type=int, default=4)
    score_plan.add_argument(
        "--token-chunk-sizes", type=int, nargs="+", default=[64, 128, 256, 512]
    )
    score_plan.add_argument(
        "--vocab-tile-sizes",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
    )
    score_plan.add_argument("--capabilities", type=Path)
    score_plan.add_argument("--output", type=Path)

    space_audit = subparsers.add_parser(
        "audit-space", help="verify and summarize a compiled search space"
    )
    space_audit.add_argument("compiled_space", type=Path)

    space_config = subparsers.add_parser(
        "space-config", help="export one deployment candidate as a calibration config"
    )
    space_config.add_argument("compiled_space", type=Path)
    space_config.add_argument("candidate_id")
    space_config.add_argument("--output", type=Path)

    candidate_plan = subparsers.add_parser(
        "plan-candidates", help="select a budgeted initial design from a compiled space"
    )
    candidate_plan.add_argument("design", type=Path)
    candidate_plan.add_argument("space", type=Path)
    candidate_plan.add_argument("compiled_space", type=Path)
    candidate_plan.add_argument("--features", type=Path)
    candidate_plan.add_argument("--output", type=Path)

    plan_audit = subparsers.add_parser(
        "audit-candidate-plan", help="verify and summarize a candidate plan"
    )
    plan_audit.add_argument("candidate_plan", type=Path)

    plan_configs = subparsers.add_parser(
        "candidate-configs", help="export selected calibration configurations"
    )
    plan_configs.add_argument("candidate_plan", type=Path)
    plan_configs.add_argument("--include-baseline", action="store_true")
    plan_configs.add_argument("--output", type=Path)

    calibration_candidates = subparsers.add_parser(
        "candidate-calibration-spec",
        help="replace manual calibration candidates with a candidate plan",
    )
    calibration_candidates.add_argument("calibration_spec", type=Path)
    calibration_candidates.add_argument("candidate_plan", type=Path)
    calibration_candidates.add_argument("--output", type=Path)

    feasibility_plan = subparsers.add_parser(
        "plan-ordered-feasibility",
        help="select the next probe for a monotone resource boundary",
    )
    feasibility_plan.add_argument("spec", type=Path)
    feasibility_plan.add_argument("--output", type=Path)

    feasibility_audit = subparsers.add_parser(
        "audit-ordered-feasibility",
        help="verify and summarize an ordered feasibility plan",
    )
    feasibility_audit.add_argument("plan", type=Path)

    policy_select = subparsers.add_parser(
        "select-policy",
        help="fit a conservative response model and emit a guarded policy bundle",
    )
    policy_select.add_argument("spec", type=Path)
    policy_select.add_argument("compiled_space", type=Path)
    policy_select.add_argument("features", type=Path)
    policy_select.add_argument("--output", type=Path)

    policy_audit = subparsers.add_parser(
        "audit-policy", help="verify and summarize a content-addressed policy bundle"
    )
    policy_audit.add_argument("policy", type=Path)

    policy_experiments = subparsers.add_parser(
        "plan-policy-experiments",
        help="choose trust-region active-learning experiments from a policy",
    )
    policy_experiments.add_argument("spec", type=Path)
    policy_experiments.add_argument("policy", type=Path)
    policy_experiments.add_argument("compiled_space", type=Path)
    policy_experiments.add_argument("features", type=Path)
    policy_experiments.add_argument("--output", type=Path)

    policy_experiment_audit = subparsers.add_parser(
        "audit-policy-experiment-plan",
        help="verify and summarize an active-learning experiment plan",
    )
    policy_experiment_audit.add_argument("plan", type=Path)

    policy_experiment_calibration = subparsers.add_parser(
        "policy-experiment-calibration-spec",
        help="replace calibration candidates with active-learning selections",
    )
    policy_experiment_calibration.add_argument("calibration_spec", type=Path)
    policy_experiment_calibration.add_argument("plan", type=Path)
    policy_experiment_calibration.add_argument(
        "--include-replay-control",
        action="store_true",
        help="prepend an identical baseline control for a local noise envelope",
    )
    policy_experiment_calibration.add_argument(
        "--campaign-id",
        help="override the template campaign id for this acquisition round",
    )
    policy_experiment_calibration.add_argument(
        "--pair-seeds",
        nargs="+",
        type=int,
        help="override pair seeds; provide exactly two seeds per protocol block",
    )

    mechanism_probe = subparsers.add_parser(
        "plan-mechanism-probes",
        help="rank legal probes from graph bindings and formal runtime pressure",
    )
    mechanism_probe.add_argument("spec", type=Path)
    mechanism_probe.add_argument("graph", type=Path)
    mechanism_probe.add_argument("compiled_space", type=Path)
    mechanism_probe.add_argument("features", type=Path)
    mechanism_probe.add_argument("--output", type=Path)

    mechanism_probe_audit = subparsers.add_parser(
        "audit-mechanism-probe-plan",
        help="verify and summarize a content-addressed mechanism probe plan",
    )
    mechanism_probe_audit.add_argument("plan", type=Path)

    mechanism_probe_calibration = subparsers.add_parser(
        "mechanism-probe-calibration-spec",
        help="compile mechanism-guided selections into the paired harness",
    )
    mechanism_probe_calibration.add_argument("calibration_spec", type=Path)
    mechanism_probe_calibration.add_argument("plan", type=Path)
    mechanism_probe_calibration.add_argument(
        "--include-replay-control",
        action="store_true",
        help="prepend an identical baseline control for a local noise envelope",
    )
    mechanism_probe_calibration.add_argument("--campaign-id")
    mechanism_probe_calibration.add_argument("--pair-seeds", nargs="+", type=int)
    mechanism_probe_calibration.add_argument("--output", type=Path)

    mechanism_probe_assessment = subparsers.add_parser(
        "assess-mechanism-probe",
        help="apply a frozen improvement gate to a completed mechanism probe",
    )
    mechanism_probe_assessment.add_argument("spec", type=Path)
    mechanism_probe_assessment.add_argument("plan", type=Path)
    mechanism_probe_assessment.add_argument("calibration_assessment", type=Path)
    mechanism_probe_assessment.add_argument("--output", type=Path)

    mechanism_probe_assessment_audit = subparsers.add_parser(
        "audit-mechanism-probe-assessment",
        help="verify and summarize a mechanism probe assessment",
    )
    mechanism_probe_assessment_audit.add_argument("assessment", type=Path)
    policy_experiment_calibration.add_argument("--output", type=Path)

    policy_experiment_assessment = subparsers.add_parser(
        "assess-policy-experiment",
        help="apply a frozen improvement gate to an active-learning experiment",
    )
    policy_experiment_assessment.add_argument("spec", type=Path)
    policy_experiment_assessment.add_argument("plan", type=Path)
    policy_experiment_assessment.add_argument("calibration_assessment", type=Path)
    policy_experiment_assessment.add_argument("--output", type=Path)

    policy_experiment_assessment_audit = subparsers.add_parser(
        "audit-policy-experiment-assessment",
        help="verify and summarize an active-learning experiment assessment",
    )
    policy_experiment_assessment_audit.add_argument("assessment", type=Path)

    execution_readiness = subparsers.add_parser(
        "assess-execution-readiness",
        help="gate a campaign using audited retry history and expected NPU cost",
    )
    execution_readiness.add_argument("spec", type=Path)
    execution_readiness.add_argument("target_plan", type=Path)
    execution_readiness.add_argument("history_campaigns", type=Path, nargs="+")
    execution_readiness.add_argument("--output", type=Path)

    execution_readiness_audit = subparsers.add_parser(
        "audit-execution-readiness",
        help="verify and summarize a content-addressed launch/defer assessment",
    )
    execution_readiness_audit.add_argument("assessment", type=Path)

    sequential_effect = subparsers.add_parser(
        "assess-sequential-effect",
        help="apply replay-adjusted anytime-valid boundaries to paired assessments",
    )
    sequential_effect.add_argument("spec", type=Path)
    sequential_effect.add_argument("calibration_assessments", type=Path, nargs="+")
    sequential_effect.add_argument("--output", type=Path)

    sequential_effect_audit = subparsers.add_parser(
        "audit-sequential-effect",
        help="verify and summarize a content-addressed sequential effect assessment",
    )
    sequential_effect_audit.add_argument("assessment", type=Path)

    sequential_design = subparsers.add_parser(
        "compare-sequential-designs",
        help="diagnose finite-budget confidence methods on frozen effect evidence",
    )
    sequential_design.add_argument("spec", type=Path)
    sequential_design.add_argument("assessment", type=Path)
    sequential_design.add_argument("--output", type=Path)

    sequential_design_audit = subparsers.add_parser(
        "audit-sequential-design-study",
        help="verify and summarize a content-addressed sequential design study",
    )
    sequential_design_audit.add_argument("study", type=Path)

    sequential_calibration = subparsers.add_parser(
        "sequential-effect-calibration-spec",
        help="emit replay-plus-candidate calibration with frozen fresh seeds",
    )
    sequential_calibration.add_argument("spec", type=Path)
    sequential_calibration.add_argument("calibration_template", type=Path)
    sequential_calibration.add_argument("--campaign-id", required=True)
    sequential_calibration.add_argument(
        "--pair-seeds", nargs="+", type=int, required=True
    )
    sequential_calibration.add_argument("--output", type=Path)

    policy_holdout = subparsers.add_parser(
        "assess-policy-holdout",
        help="compare a frozen policy with its baseline and measured holdout oracle",
    )
    policy_holdout.add_argument("spec", type=Path)
    policy_holdout.add_argument("policy", type=Path)
    policy_holdout.add_argument("compiled_space", type=Path)
    policy_holdout.add_argument("features", type=Path)
    policy_holdout.add_argument("--output", type=Path)

    policy_holdout_audit = subparsers.add_parser(
        "audit-policy-holdout",
        help="verify and summarize a policy holdout assessment",
    )
    policy_holdout_audit.add_argument("assessment", type=Path)

    policy_transfer = subparsers.add_parser(
        "plan-policy-transfer",
        help="compile a guarded selected-versus-fallback workload transfer",
    )
    policy_transfer.add_argument("spec", type=Path)
    policy_transfer.add_argument("policy", type=Path)
    policy_transfer.add_argument("calibration_template", type=Path)
    policy_transfer.add_argument("--output", type=Path)

    policy_transfer_audit = subparsers.add_parser(
        "audit-policy-transfer",
        help="verify and summarize a content-addressed policy transfer plan",
    )
    policy_transfer_audit.add_argument("plan", type=Path)

    policy_transfer_calibration = subparsers.add_parser(
        "policy-transfer-calibration-spec",
        help="export the runnable calibration embedded in a policy transfer plan",
    )
    policy_transfer_calibration.add_argument("plan", type=Path)
    policy_transfer_calibration.add_argument("--output", type=Path)

    policy_transfer_assessment = subparsers.add_parser(
        "assess-policy-transfer",
        help="separate positive transfer evidence from source-policy activation",
    )
    policy_transfer_assessment.add_argument("spec", type=Path)
    policy_transfer_assessment.add_argument("plan", type=Path)
    policy_transfer_assessment.add_argument("calibration_assessment", type=Path)
    policy_transfer_assessment.add_argument("--output", type=Path)

    policy_transfer_assessment_audit = subparsers.add_parser(
        "audit-policy-transfer-assessment",
        help="verify and summarize a policy transfer assessment",
    )
    policy_transfer_assessment_audit.add_argument("assessment", type=Path)

    runtime_pool = subparsers.add_parser(
        "compile-runtime-policy-pool",
        help="bind independently validated policies to prewarmed endpoints",
    )
    runtime_pool.add_argument("spec", type=Path)
    runtime_pool.add_argument("--policy", type=Path, action="append", required=True)
    runtime_pool.add_argument("--assessment", type=Path, action="append", required=True)
    runtime_pool.add_argument("--output", type=Path)

    runtime_pool_audit = subparsers.add_parser(
        "audit-runtime-policy-pool",
        help="verify and summarize a content-addressed runtime policy pool",
    )
    runtime_pool_audit.add_argument("pool", type=Path)

    runtime_route = subparsers.add_parser(
        "route-runtime-policy",
        help="route one request group through validated guards or fallback",
    )
    runtime_route.add_argument("pool", type=Path)
    runtime_route.add_argument("request", type=Path)
    runtime_route.add_argument("--output", type=Path)

    runtime_decision_audit = subparsers.add_parser(
        "audit-runtime-policy-decision",
        help="verify and summarize a content-addressed runtime routing decision",
    )
    runtime_decision_audit.add_argument("decision", type=Path)

    capacity_graph_repair = subparsers.add_parser(
        "plan-capacity-graph-repair",
        help="diagnose measured scheduler/graph-domain conflicts and plan a repair",
    )
    capacity_graph_repair.add_argument("assessment", type=Path)
    capacity_graph_repair.add_argument("calibration_spec", type=Path)
    capacity_graph_repair.add_argument("--repair-id", required=True)
    capacity_graph_repair.add_argument("--candidate-configuration-id")
    capacity_graph_repair.add_argument("--output", type=Path)

    capacity_graph_repair_audit = subparsers.add_parser(
        "audit-capacity-graph-repair",
        help="verify and summarize a content-addressed capacity/graph repair plan",
    )
    capacity_graph_repair_audit.add_argument("plan", type=Path)

    capacity_graph_repair_calibration = subparsers.add_parser(
        "capacity-graph-repair-calibration-spec",
        help="compile a capacity/graph repair into a paired calibration spec",
    )
    capacity_graph_repair_calibration.add_argument("calibration_spec", type=Path)
    capacity_graph_repair_calibration.add_argument("plan", type=Path)
    capacity_graph_repair_calibration.add_argument("--campaign-id", required=True)
    capacity_graph_repair_calibration.add_argument(
        "--pair-seeds", type=int, nargs="+", required=True
    )
    capacity_graph_repair_calibration.add_argument(
        "--without-replay-control", action="store_true"
    )
    capacity_graph_repair_calibration.add_argument("--output", type=Path)

    capture_plan = subparsers.add_parser(
        "plan-graph-capture",
        help="select stage-aware CUDA or ACL graph capture buckets",
    )
    capture_plan.add_argument("profile", type=Path)
    capture_plan.add_argument("--output", type=Path)

    capture_audit = subparsers.add_parser(
        "audit-graph-capture", help="verify and summarize a graph capture plan"
    )
    capture_audit.add_argument("plan", type=Path)

    graph_trace = subparsers.add_parser(
        "trace-graph-workload",
        help="extract stage-labelled graph shapes from a chang pressure result",
    )
    graph_trace.add_argument("result", type=Path)
    graph_trace.add_argument("--trace-id", required=True)
    graph_trace.add_argument("--output", type=Path)

    graph_trace_audit = subparsers.add_parser(
        "audit-graph-trace", help="verify and summarize a graph workload trace"
    )
    graph_trace_audit.add_argument("trace", type=Path)

    graph_metrics = subparsers.add_parser(
        "import-vllm-graph-metrics",
        help="parse role-delimited vLLM graph statistics from a profiler log",
    )
    graph_metrics.add_argument("log", type=Path)
    graph_metrics.add_argument("--profile-id", required=True)
    graph_metrics.add_argument("--output", type=Path)

    graph_metrics_audit = subparsers.add_parser(
        "audit-vllm-graph-metrics", help="verify and summarize vLLM graph metrics"
    )
    graph_metrics_audit.add_argument("profile", type=Path)

    graph_metrics_diagnosis = subparsers.add_parser(
        "diagnose-vllm-graph-runtime",
        help="classify phase-aware vLLM graph misses by execution mechanism",
    )
    graph_metrics_diagnosis.add_argument("profile", type=Path)
    graph_metrics_diagnosis.add_argument("--output", type=Path)

    stage_wavefront = subparsers.add_parser(
        "plan-stage-wavefront",
        help="plan graph-eligible admission waves for staged generation",
    )
    stage_wavefront.add_argument("--outer-concurrency", type=int, required=True)
    stage_wavefront.add_argument("--sequences-per-group", type=int, required=True)
    stage_wavefront.add_argument("--graph-capture-ceiling", type=int, required=True)
    stage_wavefront.add_argument("--scheduler-sequence-cap", type=int, required=True)
    stage_wavefront.add_argument(
        "--minimum-full-wave-utilization", type=float, default=0.65
    )
    stage_wavefront.add_argument("--output", type=Path)

    graph_metrics_merge = subparsers.add_parser(
        "merge-vllm-graph-metrics",
        help="aggregate compatible graph-shape profiles across workload regimes",
    )
    graph_metrics_merge.add_argument("profiles", type=Path, nargs="+")
    graph_metrics_merge.add_argument("--profile-id", required=True)
    graph_metrics_merge.add_argument("--output", type=Path)

    graph_corpus = subparsers.add_parser(
        "build-vllm-graph-corpus",
        help="bind train and holdout graph profiles to named workload regimes",
    )
    graph_corpus.add_argument("--corpus-id", required=True)
    graph_corpus.add_argument(
        "--train",
        action="append",
        required=True,
        metavar="REGIME_ID=PROFILE.json",
    )
    graph_corpus.add_argument(
        "--holdout",
        action="append",
        required=True,
        metavar="REGIME_ID=PROFILE.json",
    )
    graph_corpus.add_argument("--output", type=Path)

    graph_corpus_audit = subparsers.add_parser(
        "audit-vllm-graph-corpus",
        help="verify and summarize a regime-labelled graph profile corpus",
    )
    graph_corpus_audit.add_argument("corpus", type=Path)

    robust_graph = subparsers.add_parser(
        "plan-vllm-robust-graph-experiment",
        help="promote a train-derived graph policy only after holdout validation",
    )
    robust_graph.add_argument("corpus", type=Path)
    robust_graph.add_argument("--campaign-id", required=True)
    robust_graph.add_argument("--minimum-train-regimes", type=int, default=2)
    robust_graph.add_argument("--minimum-holdout-regimes", type=int, default=1)
    robust_graph.add_argument("--blocks", type=int, default=2)
    robust_graph.add_argument("--pair-seeds", type=int, nargs="+", required=True)
    robust_graph.add_argument(
        "--exclude-replay-control",
        action="store_true",
        help="omit the default-vs-default noise-control group",
    )
    robust_graph.add_argument("--promotion-output", type=Path, required=True)
    robust_graph.add_argument("--output", type=Path, required=True)

    coverage_graph = subparsers.add_parser(
        "plan-vllm-coverage-graph-experiment",
        help="search bucket subsets under train constraints and gate on holdouts",
    )
    coverage_graph.add_argument("corpus", type=Path)
    coverage_graph.add_argument("--campaign-id", required=True)
    coverage_graph.add_argument("--policy-id", default="coverage-constrained")
    coverage_graph.add_argument(
        "--candidate-engine-role",
        action="append",
        choices=("base", "proposal"),
        help="optimize only this engine role; repeat to select both",
    )
    coverage_graph.add_argument("--minimum-train-regimes", type=int, default=2)
    coverage_graph.add_argument("--minimum-holdout-regimes", type=int, default=1)
    coverage_graph.add_argument(
        "--minimum-retained-graph-fraction", type=float, default=1.0
    )
    coverage_graph.add_argument(
        "--maximum-remapped-graph-fraction", type=float, required=True
    )
    coverage_graph.add_argument(
        "--maximum-added-padding-ratio", type=float, required=True
    )
    coverage_graph.add_argument(
        "--holdout-minimum-retained-graph-fraction",
        type=float,
        help="holdout limit; defaults to the corresponding train limit",
    )
    coverage_graph.add_argument(
        "--holdout-maximum-remapped-graph-fraction",
        type=float,
        help="holdout limit; defaults to the corresponding train limit",
    )
    coverage_graph.add_argument(
        "--holdout-maximum-added-padding-ratio",
        type=float,
        help="holdout limit; defaults to the corresponding train limit",
    )
    coverage_graph.add_argument("--blocks", type=int, default=2)
    coverage_graph.add_argument("--pair-seeds", type=int, nargs="+", required=True)
    coverage_graph.add_argument(
        "--exclude-replay-control",
        action="store_true",
        help="omit the default-vs-default noise-control group",
    )
    coverage_graph.add_argument("--search-output", type=Path, required=True)
    coverage_graph.add_argument("--output", type=Path, required=True)

    graph_experiment = subparsers.add_parser(
        "plan-vllm-graph-experiment",
        help="build default/pruned/no-graph ABBA runs from measured graph shapes",
    )
    graph_experiment.add_argument("profile", type=Path)
    graph_experiment.add_argument("--campaign-id", required=True)
    graph_experiment.add_argument("--blocks", type=int, default=1)
    graph_experiment.add_argument("--pair-seeds", type=int, nargs="+", required=True)
    graph_experiment.add_argument(
        "--include-replay-control",
        action="store_true",
        help="prepend an identical-default ABBA group to measure harness nondeterminism",
    )
    graph_experiment.add_argument("--output", type=Path)

    graph_experiment_audit = subparsers.add_parser(
        "audit-vllm-graph-experiment",
        help="verify and summarize a graph experiment plan",
    )
    graph_experiment_audit.add_argument("plan", type=Path)

    graph_calibration = subparsers.add_parser(
        "build-vllm-graph-calibration",
        help="bind graph policies to a full calibration workload and environment",
    )
    graph_calibration.add_argument("template", type=Path)
    graph_calibration.add_argument("graph_plan", type=Path)
    graph_calibration.add_argument("--campaign-id", required=True)
    graph_calibration.add_argument("--candidate-policy-id", action="append")
    graph_calibration.add_argument("--include-replay-control", action="store_true")
    graph_calibration.add_argument("--output", type=Path)

    graph_coverage = subparsers.add_parser(
        "evaluate-vllm-graph-policy",
        help="measure graph remapping and padding on an independent shape profile",
    )
    graph_coverage.add_argument("graph_plan", type=Path)
    graph_coverage.add_argument("profile", type=Path)
    graph_coverage.add_argument("--policy-id", required=True)
    graph_coverage.add_argument("--output", type=Path)

    graph_assessment = subparsers.add_parser(
        "assess-vllm-graph-experiment",
        help="audit completed graph-policy runs and compute paired effects",
    )
    graph_assessment.add_argument("plan", type=Path)
    graph_assessment.add_argument("results", type=Path)
    graph_assessment.add_argument("--comparison-id", required=True)
    graph_assessment.add_argument("--minimum-requests", type=int, default=32)
    graph_assessment.add_argument("--minimum-blocks", type=int, default=2)
    graph_assessment.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "graph":
            graph = build_conditional_is_small_proposal_graph(
                candidate_count=args.candidate_count,
                rollout_count=args.rollout_count,
                block_size=args.block_size,
                total_length=args.total_length,
                apply_importance_correction=not args.disable_importance_correction,
            )
            _write_json(graph.to_dict(), args.output)
            return 0
        if args.command == "import-results":
            ledger = import_results(args.source)
            _write_json(ledger.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(ledger.audit(), sort_keys=True) + "\n")
            return 2 if args.strict and ledger.rejections else 0
        if args.command == "audit":
            _write_json(_load_ledger(args.ledger).audit(), None)
            return 0
        if args.command == "merge-ledgers":
            ledger = merge_evidence_ledgers(
                tuple(_load_ledger(path) for path in args.ledgers)
            )
            _write_json(ledger.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(ledger.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "plan-calibration":
            spec = CalibrationSpec.from_dict(
                _load_object(args.spec, "calibration spec")
            )
            _write_json(build_plan(spec).to_dict(), args.output)
            return 0
        if args.command == "run-manifest":
            plan = CalibrationPlan.from_dict(
                _load_object(args.plan, "calibration plan")
            )
            _write_json(build_run_manifest(plan, args.run_id), args.output)
            return 0
        if args.command == "assess-calibration":
            plan = CalibrationPlan.from_dict(
                _load_object(args.plan, "calibration plan")
            )
            noise_reference = (
                None
                if args.replay_noise_assessment is None
                else load_replay_noise_reference(args.replay_noise_assessment)
            )
            assessment = assess_calibration(
                plan,
                load_observations(args.observations),
                replay_noise_reference=noise_reference,
            )
            _write_json(assessment.to_dict(), args.output)
            return 0 if assessment.formal_complete else 2
        if args.command == "assess-harness-cost":
            plan = CalibrationPlan.from_dict(
                _load_object(args.plan, "calibration plan")
            )
            assessment = assess_harness_cost(plan, args.campaign_root)
            _write_json(assessment.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(assessment.audit(), sort_keys=True) + "\n")
            return 0 if assessment.complete else 2
        if args.command == "audit-harness-cost":
            assessment = HarnessCostAssessment.from_dict(
                _load_object(args.assessment, "harness cost assessment")
            )
            _write_json(assessment.audit(), None)
            return 0 if assessment.complete else 2
        if args.command == "attest-runtime-closure":
            plan = CalibrationPlan.from_dict(
                _load_object(args.plan, "calibration plan")
            )
            assessment = attest_runtime_closure(plan, args.campaign_root)
            _write_json(assessment.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(assessment.audit(), sort_keys=True) + "\n")
            return 0 if assessment.complete else 2
        if args.command == "audit-runtime-closure":
            assessment = RuntimeClosureAssessment.from_dict(
                _load_object(args.assessment, "runtime closure assessment")
            )
            _write_json(assessment.audit(), None)
            return 0 if assessment.complete else 2
        if args.command == "enrich-runtime-features":
            table = _load_feature_table(args.features)
            assessment = RuntimeClosureAssessment.from_dict(
                _load_object(args.assessment, "runtime closure assessment")
            )
            enriched = enrich_features_with_runtime_closure(table, assessment)
            _write_json(enriched.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(enriched.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "plan-engine-lifecycle":
            spec = EngineLifecyclePlanningSpec.from_dict(
                _load_object(args.spec, "engine lifecycle planning spec")
            )
            plan = CalibrationPlan.from_dict(
                _load_object(args.plan, "calibration plan")
            )
            assessment = HarnessCostAssessment.from_dict(
                _load_object(
                    args.harness_cost_assessment, "harness cost assessment"
                )
            )
            lifecycle = plan_engine_lifecycle(spec, plan, assessment)
            _write_json(lifecycle.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(lifecycle.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "audit-engine-lifecycle":
            lifecycle = EngineLifecyclePlan.from_dict(
                _load_object(args.lifecycle_plan, "engine lifecycle plan")
            )
            _write_json(lifecycle.audit(), None)
            return 0
        if args.command == "audit-lifecycle-execution":
            lifecycle = EngineLifecyclePlan.from_dict(
                _load_object(args.lifecycle_plan, "engine lifecycle plan")
            )
            receipt = LifecycleExecutionReceipt.from_dict(
                _load_object(args.execution_receipt, "lifecycle execution receipt")
            )
            _write_json(audit_lifecycle_execution(lifecycle, receipt), None)
            return 0
        if args.command == "prepare-run":
            plan = CalibrationPlan.from_dict(
                _load_object(args.plan, "calibration plan")
            )
            bundle = build_chang_run_bundle(
                plan,
                args.run_id,
                source_repo=args.source_repo,
                source_config=args.source_config,
                data=args.data,
                output_dir=args.output_dir,
                python_executable=args.python,
            )
            bundle.write(args.output_dir)
            compatibility = bundle.launch["compatibility"]
            sys.stdout.write(json.dumps(compatibility, sort_keys=True) + "\n")
            return 2 if args.require_formal and not bundle.formal_eligible else 0
        if args.command == "observe-run":
            observation = observation_from_chang_result(
                load_manifest(args.manifest), args.native_result
            )
            _write_json(observation.to_dict(), args.output)
            return 0
        if args.command == "features-ledger":
            table = features_from_ledger(_load_ledger(args.ledger))
            _write_json(table.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(table.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "features-run":
            observation = RunObservation.from_dict(
                _load_object(args.observation, "run observation")
            )
            table = features_from_run(load_manifest(args.manifest), observation)
            _write_json(table.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(table.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "audit-features":
            _write_json(_load_feature_table(args.table).audit(), None)
            return 0
        if args.command == "fit-npu-cost-model":
            corpus = NPUCalibrationCorpus.from_dict(
                _load_object(args.calibration, "NPU calibration corpus")
            )
            model = fit_npu_stage_cost_model(corpus)
            _write_json(model.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(
                    json.dumps(
                        {
                            "model_sha256": model.sha256,
                            "stage_count": len(model.stage_metadata),
                            "bucket_count": len(model.fits),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            return 0
        if args.command == "predict-npu-graph-cost":
            model = NPUStageCostModel.from_dict(
                _load_object(args.model, "NPU stage cost model")
            )
            query = stage_queries_from_dict(
                _load_object(args.query, "NPU graph cost query")
            )
            prediction = predict_serial_graph_cost(model, query)
            _write_json(prediction.to_dict(), args.output)
            return 0 if prediction.status == "supported" else 2
        if args.command == "compile-space":
            space = DeploymentSearchSpace.from_dict(
                _load_object(args.space, "deployment search space")
            )
            if args.graph is not None:
                graph = InferenceGraph.from_dict(
                    _load_object(args.graph, "inference graph")
                )
                space.validate_against_graph(graph)
            compiled = compile_search_space(
                space,
                capability_profile=(
                    RuntimeCapabilityProfile.from_dict(
                        _load_object(args.capabilities, "runtime capability profile")
                    )
                    if args.capabilities is not None
                    else None
                ),
                allow_algorithm_changes=args.allow_algorithm_changes,
                max_cartesian_product=args.max_cartesian_product,
            )
            _write_json(compiled.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(compiled.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "audit-space":
            _write_json(_load_compiled_space(args.compiled_space).audit(), None)
            return 0
        if args.command == "plan-score-reduction":
            capability_profile = (
                RuntimeCapabilityProfile.from_dict(
                    _load_object(args.capabilities, "runtime capability profile")
                )
                if args.capabilities is not None
                else None
            )
            plan = build_score_reduction_plan(
                requirement_for_reward(
                    args.reward,
                    importance_correction=args.importance_correction,
                    consilience_top_k=args.consilience_top_k,
                ),
                ScoreWorkload(
                    positions=args.positions,
                    vocab_size=args.vocab_size,
                    logits_element_bytes=args.logits_element_bytes,
                    accumulator_bytes=args.accumulator_bytes,
                ),
                memory_budget_bytes=int(args.memory_budget_gib * 1024**3),
                capability_profile=capability_profile,
                token_chunk_sizes=args.token_chunk_sizes,
                vocab_tile_sizes=args.vocab_tile_sizes,
            )
            _write_json(plan, args.output)
            return 0
        if args.command == "space-config":
            compiled = _load_compiled_space(args.compiled_space)
            configuration = configuration_from_candidate(compiled, args.candidate_id)
            _write_json(configuration.to_dict(), args.output)
            return 0
        if args.command == "plan-candidates":
            design = CandidateDesignSpec.from_dict(
                _load_object(args.design, "candidate design spec")
            )
            space = DeploymentSearchSpace.from_dict(
                _load_object(args.space, "deployment search space")
            )
            compiled = _load_compiled_space(args.compiled_space)
            features = (
                None if args.features is None else _load_feature_table(args.features)
            )
            plan = build_candidate_plan(design, space, compiled, features)
            _write_json(plan.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(plan.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "audit-candidate-plan":
            _write_json(_load_candidate_plan(args.candidate_plan).audit(), None)
            return 0
        if args.command == "candidate-configs":
            plan = _load_candidate_plan(args.candidate_plan)
            configurations = configurations_from_plan(
                plan, include_baseline=args.include_baseline
            )
            payload = {
                "schema_version": "1.0",
                "candidate_plan_sha256": plan.to_dict()["candidate_plan_sha256"],
                "configurations": [
                    configuration.to_dict() for configuration in configurations
                ],
            }
            _write_json(payload, args.output)
            return 0
        if args.command == "candidate-calibration-spec":
            spec = CalibrationSpec.from_dict(
                _load_object(args.calibration_spec, "calibration spec")
            )
            plan = _load_candidate_plan(args.candidate_plan)
            updated = calibration_spec_from_candidate_plan(spec, plan)
            _write_json(updated.to_dict(), args.output)
            return 0
        if args.command == "plan-ordered-feasibility":
            spec = OrderedFeasibilitySpec.from_dict(
                _load_object(args.spec, "ordered feasibility spec")
            )
            plan = plan_ordered_feasibility(spec)
            _write_json(plan.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(plan.audit(), sort_keys=True) + "\n")
            return 0 if plan.eligible else 2
        if args.command == "audit-ordered-feasibility":
            _write_json(_load_ordered_feasibility_plan(args.plan).audit(), None)
            return 0
        if args.command == "select-policy":
            spec = PolicySelectionSpec.from_dict(
                _load_object(args.spec, "policy selection spec")
            )
            bundle = select_policy(
                spec,
                _load_compiled_space(args.compiled_space),
                _load_feature_table(args.features),
            )
            _write_json(bundle.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(bundle.audit(), sort_keys=True) + "\n")
            return 0 if bundle.status == "selected" else 2
        if args.command == "audit-policy":
            bundle = PolicyBundle.from_dict(_load_object(args.policy, "policy bundle"))
            _write_json(bundle.audit(), None)
            return 0 if bundle.status == "selected" else 2
        if args.command == "plan-policy-experiments":
            spec = PolicyAcquisitionSpec.from_dict(
                _load_object(args.spec, "policy acquisition spec")
            )
            policy = PolicyBundle.from_dict(_load_object(args.policy, "policy bundle"))
            plan = plan_policy_experiments(
                spec,
                policy,
                _load_compiled_space(args.compiled_space),
                _load_feature_table(args.features),
            )
            _write_json(plan.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(plan.audit(), sort_keys=True) + "\n")
            return 0 if plan.status == "planned" else 2
        if args.command == "audit-policy-experiment-plan":
            plan = PolicyExperimentPlan.from_dict(
                _load_object(args.plan, "policy experiment plan")
            )
            _write_json(plan.audit(), None)
            return 0 if plan.status == "planned" else 2
        if args.command == "policy-experiment-calibration-spec":
            calibration_spec = CalibrationSpec.from_dict(
                _load_object(args.calibration_spec, "calibration spec")
            )
            plan = PolicyExperimentPlan.from_dict(
                _load_object(args.plan, "policy experiment plan")
            )
            updated = calibration_spec_from_policy_experiment_plan(
                calibration_spec,
                plan,
                include_replay_control=args.include_replay_control,
                campaign_id=args.campaign_id,
                pair_seeds=args.pair_seeds,
            )
            _write_json(updated.to_dict(), args.output)
            return 0
        if args.command == "assess-policy-experiment":
            spec = PolicyExperimentAssessmentSpec.from_dict(
                _load_object(args.spec, "policy experiment assessment spec")
            )
            plan = PolicyExperimentPlan.from_dict(
                _load_object(args.plan, "policy experiment plan")
            )
            assessment = assess_policy_experiment(
                spec,
                plan,
                _load_object(args.calibration_assessment, "calibration assessment"),
            )
            _write_json(assessment.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(assessment.audit(), sort_keys=True) + "\n")
            return 0 if assessment.status == "validated_signal" else 2
        if args.command == "audit-policy-experiment-assessment":
            assessment = PolicyExperimentAssessment.from_dict(
                _load_object(args.assessment, "policy experiment assessment")
            )
            _write_json(assessment.audit(), None)
            return 0 if assessment.status == "validated_signal" else 2
        if args.command == "assess-execution-readiness":
            spec = ExecutionReadinessSpec.from_dict(
                _load_object(args.spec, "execution-readiness spec")
            )
            target_plan = CalibrationPlan.from_dict(
                _load_object(args.target_plan, "target calibration plan")
            )
            assessment = assess_execution_readiness(
                spec,
                target_plan,
                args.target_plan.expanduser().resolve(),
                [path.expanduser().resolve() for path in args.history_campaigns],
            )
            _write_json(assessment.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(assessment.audit(), sort_keys=True) + "\n")
            return 0 if assessment.status == "launch" else 2
        if args.command == "audit-execution-readiness":
            assessment = ExecutionReadinessAssessment.from_dict(
                _load_object(args.assessment, "execution-readiness assessment")
            )
            _write_json(assessment.audit(), None)
            return 0 if assessment.status == "launch" else 2
        if args.command == "assess-sequential-effect":
            spec = SequentialEffectSpec.from_dict(
                _load_object(args.spec, "sequential effect spec")
            )
            assessment = assess_sequential_effect_files(
                spec,
                [
                    str(path.expanduser().resolve())
                    for path in args.calibration_assessments
                ],
            )
            _write_json(assessment.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(assessment.audit(), sort_keys=True) + "\n")
            return 0 if assessment.status in {"promote", "close_direction"} else 2
        if args.command == "audit-sequential-effect":
            assessment = SequentialEffectAssessment.from_dict(
                _load_object(args.assessment, "sequential effect assessment")
            )
            _write_json(assessment.audit(), None)
            return 0 if assessment.status in {"promote", "close_direction"} else 2
        if args.command == "compare-sequential-designs":
            study_spec = SequentialDesignStudySpec.from_dict(
                _load_object(args.spec, "sequential design study spec")
            )
            assessment = SequentialEffectAssessment.from_dict(
                _load_object(args.assessment, "sequential effect assessment")
            )
            study = build_sequential_design_study(study_spec, assessment)
            _write_json(study.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(study.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "audit-sequential-design-study":
            study = SequentialDesignStudy.from_dict(
                _load_object(args.study, "sequential design study")
            )
            _write_json(study.audit(), None)
            return 0
        if args.command == "sequential-effect-calibration-spec":
            spec = SequentialEffectSpec.from_dict(
                _load_object(args.spec, "sequential effect spec")
            )
            template = CalibrationSpec.from_dict(
                _load_object(args.calibration_template, "calibration template")
            )
            updated = calibration_spec_from_sequential_effect(
                template,
                spec,
                args.campaign_id,
                args.pair_seeds,
            )
            _write_json(updated.to_dict(), args.output)
            return 0
        if args.command == "plan-mechanism-probes":
            spec = MechanismProbeSpec.from_dict(
                _load_object(args.spec, "mechanism probe spec")
            )
            graph = InferenceGraph.from_dict(
                _load_object(args.graph, "inference graph")
            )
            plan = plan_mechanism_probes(
                spec,
                graph,
                _load_compiled_space(args.compiled_space),
                _load_feature_table(args.features),
            )
            _write_json(plan.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(plan.audit(), sort_keys=True) + "\n")
            return 0 if plan.status == "planned" else 2
        if args.command == "audit-mechanism-probe-plan":
            plan = MechanismProbePlan.from_dict(
                _load_object(args.plan, "mechanism probe plan")
            )
            _write_json(plan.audit(), None)
            return 0 if plan.status == "planned" else 2
        if args.command == "mechanism-probe-calibration-spec":
            calibration_spec = CalibrationSpec.from_dict(
                _load_object(args.calibration_spec, "calibration spec")
            )
            plan = MechanismProbePlan.from_dict(
                _load_object(args.plan, "mechanism probe plan")
            )
            updated = calibration_spec_from_mechanism_probe_plan(
                calibration_spec,
                plan,
                include_replay_control=args.include_replay_control,
                campaign_id=args.campaign_id,
                pair_seeds=args.pair_seeds,
            )
            _write_json(updated.to_dict(), args.output)
            return 0
        if args.command == "assess-mechanism-probe":
            spec = MechanismProbeAssessmentSpec.from_dict(
                _load_object(args.spec, "mechanism probe assessment spec")
            )
            plan = MechanismProbePlan.from_dict(
                _load_object(args.plan, "mechanism probe plan")
            )
            assessment = assess_mechanism_probe(
                spec,
                plan,
                _load_object(args.calibration_assessment, "calibration assessment"),
            )
            _write_json(assessment.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(assessment.audit(), sort_keys=True) + "\n")
            return 0 if assessment.status == "validated_signal" else 2
        if args.command == "audit-mechanism-probe-assessment":
            assessment = MechanismProbeAssessment.from_dict(
                _load_object(args.assessment, "mechanism probe assessment")
            )
            _write_json(assessment.audit(), None)
            return 0 if assessment.status == "validated_signal" else 2
        if args.command == "assess-policy-holdout":
            spec = PolicyHoldoutSpec.from_dict(
                _load_object(args.spec, "policy holdout spec")
            )
            policy = PolicyBundle.from_dict(_load_object(args.policy, "policy bundle"))
            assessment = assess_policy_holdout(
                spec,
                policy,
                _load_compiled_space(args.compiled_space),
                _load_feature_table(args.features),
            )
            _write_json(assessment.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(assessment.audit(), sort_keys=True) + "\n")
            return 0 if assessment.status == "validated" else 2
        if args.command == "audit-policy-holdout":
            assessment = PolicyHoldoutAssessment.from_dict(
                _load_object(args.assessment, "policy holdout assessment")
            )
            _write_json(assessment.audit(), None)
            return 0 if assessment.status == "validated" else 2
        if args.command == "plan-policy-transfer":
            transfer_spec = PolicyTransferSpec.from_dict(
                _load_object(args.spec, "policy transfer spec")
            )
            policy = PolicyBundle.from_dict(_load_object(args.policy, "policy bundle"))
            calibration_template = CalibrationSpec.from_dict(
                _load_object(args.calibration_template, "calibration template")
            )
            plan = plan_policy_transfer(transfer_spec, policy, calibration_template)
            _write_json(plan.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(plan.audit(), sort_keys=True) + "\n")
            return 0 if plan.status == "planned" else 2
        if args.command == "audit-policy-transfer":
            plan = PolicyTransferPlan.from_dict(
                _load_object(args.plan, "policy transfer plan")
            )
            _write_json(plan.audit(), None)
            return 0 if plan.status == "planned" else 2
        if args.command == "policy-transfer-calibration-spec":
            plan = PolicyTransferPlan.from_dict(
                _load_object(args.plan, "policy transfer plan")
            )
            calibration = calibration_spec_from_policy_transfer_plan(plan)
            _write_json(calibration.to_dict(), args.output)
            return 0
        if args.command == "assess-policy-transfer":
            spec = PolicyTransferAssessmentSpec.from_dict(
                _load_object(args.spec, "policy transfer assessment spec")
            )
            plan = PolicyTransferPlan.from_dict(
                _load_object(args.plan, "policy transfer plan")
            )
            assessment = assess_policy_transfer(
                spec,
                plan,
                _load_object(args.calibration_assessment, "calibration assessment"),
            )
            _write_json(assessment.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(assessment.audit(), sort_keys=True) + "\n")
            return 0 if assessment.status == "positive_transfer_evidence" else 2
        if args.command == "audit-policy-transfer-assessment":
            assessment = PolicyTransferAssessment.from_dict(
                _load_object(args.assessment, "policy transfer assessment")
            )
            _write_json(assessment.audit(), None)
            return 0 if assessment.status == "positive_transfer_evidence" else 2
        if args.command == "compile-runtime-policy-pool":
            spec = RuntimePolicyPoolSpec.from_dict(
                _load_object(args.spec, "runtime policy pool spec")
            )
            policies = [
                PolicyBundle.from_dict(_load_object(path, "runtime policy bundle"))
                for path in args.policy
            ]
            assessments = [
                PolicyHoldoutAssessment.from_dict(
                    _load_object(path, "runtime policy holdout assessment")
                )
                for path in args.assessment
            ]
            pool = compile_runtime_policy_pool(spec, policies, assessments)
            _write_json(pool.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(pool.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "audit-runtime-policy-pool":
            pool = RuntimePolicyPool.from_dict(
                _load_object(args.pool, "runtime policy pool")
            )
            _write_json(pool.audit(), None)
            return 0
        if args.command == "route-runtime-policy":
            pool = RuntimePolicyPool.from_dict(
                _load_object(args.pool, "runtime policy pool")
            )
            request = RuntimeRoutingRequest.from_dict(
                _load_object(args.request, "runtime routing request")
            )
            decision = route_runtime_policy(pool, request)
            _write_json(decision.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(decision.audit(), sort_keys=True) + "\n")
            return 0 if decision.status == "selected" else 2
        if args.command == "audit-runtime-policy-decision":
            decision = RuntimePolicyDecision.from_dict(
                _load_object(args.decision, "runtime policy decision")
            )
            _write_json(decision.audit(), None)
            return 0 if decision.status == "selected" else 2
        if args.command == "plan-capacity-graph-repair":
            calibration_spec = CalibrationSpec.from_dict(
                _load_object(args.calibration_spec, "calibration spec")
            )
            plan = plan_capacity_graph_repair(
                _load_object(args.assessment, "calibration assessment"),
                calibration_spec,
                repair_id=args.repair_id,
                candidate_configuration_id=args.candidate_configuration_id,
            )
            _write_json(plan.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(plan.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "audit-capacity-graph-repair":
            plan = CapacityGraphRepairPlan.from_dict(
                _load_object(args.plan, "capacity-graph repair plan")
            )
            _write_json(plan.audit(), None)
            return 0
        if args.command == "capacity-graph-repair-calibration-spec":
            calibration_spec = CalibrationSpec.from_dict(
                _load_object(args.calibration_spec, "calibration spec")
            )
            plan = CapacityGraphRepairPlan.from_dict(
                _load_object(args.plan, "capacity-graph repair plan")
            )
            updated = calibration_spec_from_capacity_graph_repair(
                calibration_spec,
                plan,
                campaign_id=args.campaign_id,
                pair_seeds=args.pair_seeds,
                include_replay_control=not args.without_replay_control,
            )
            _write_json(updated.to_dict(), args.output)
            return 0
        if args.command == "plan-graph-capture":
            spec = CapturePlanningSpec.from_dict(
                _load_object(args.profile, "capture planning spec")
            )
            plan = plan_graph_capture(spec)
            _write_json(plan.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(plan.metrics, sort_keys=True) + "\n")
            return 0
        if args.command == "audit-graph-capture":
            plan = _load_capture_plan(args.plan)
            _write_json(
                {
                    "profile_id": plan.profile_id,
                    "selected_capture_sizes": list(plan.selected_capture_sizes),
                    "metrics": dict(sorted(plan.metrics.items())),
                },
                None,
            )
            return 0
        if args.command == "trace-graph-workload":
            result = _load_object(args.result, "chang pressure result")
            trace = trace_from_chang_result(result, trace_id=args.trace_id)
            _write_json(trace.artifact_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(trace.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "audit-graph-trace":
            trace = GraphWorkloadTrace.from_dict(
                _load_object(args.trace, "graph workload trace")
            )
            _write_json(trace.audit(), None)
            return 0
        if args.command == "import-vllm-graph-metrics":
            log_bytes = args.log.expanduser().read_bytes()
            log_text = log_bytes.decode("utf-8")
            profile = parse_vllm_graph_metrics(
                log_text,
                profile_id=args.profile_id,
                source_log_sha256=hashlib.sha256(log_bytes).hexdigest(),
            )
            _write_json(profile.artifact_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(profile.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "audit-vllm-graph-metrics":
            profile = VLLMGraphMetricsProfile.from_dict(
                _load_object(args.profile, "vLLM graph metrics profile")
            )
            _write_json(profile.audit(), None)
            return 0
        if args.command == "diagnose-vllm-graph-runtime":
            profile = VLLMGraphMetricsProfile.from_dict(
                _load_object(args.profile, "vLLM graph metrics profile")
            )
            diagnosis = profile.diagnose()
            _write_json(diagnosis, args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(diagnosis, sort_keys=True) + "\n")
            return 0
        if args.command == "plan-stage-wavefront":
            plan = plan_stage_wavefront(
                outer_concurrency=args.outer_concurrency,
                sequences_per_group=args.sequences_per_group,
                graph_capture_ceiling=args.graph_capture_ceiling,
                scheduler_sequence_cap=args.scheduler_sequence_cap,
                minimum_full_wave_utilization=(
                    args.minimum_full_wave_utilization
                ),
            )
            _write_json(plan.to_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(
                    json.dumps(plan.selected.to_dict(), sort_keys=True) + "\n"
                )
            return 0
        if args.command == "merge-vllm-graph-metrics":
            profiles = [
                VLLMGraphMetricsProfile.from_dict(
                    _load_object(path, "vLLM graph metrics profile")
                )
                for path in args.profiles
            ]
            profile = merge_vllm_graph_metrics(
                profiles,
                profile_id=args.profile_id,
            )
            _write_json(profile.artifact_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(profile.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "build-vllm-graph-corpus":
            corpus = build_graph_profile_corpus(
                corpus_id=args.corpus_id,
                train_profiles=_load_regime_profiles(args.train),
                holdout_profiles=_load_regime_profiles(args.holdout),
            )
            _write_json(corpus.artifact_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(corpus.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "audit-vllm-graph-corpus":
            corpus = GraphProfileCorpus.from_dict(
                _load_object(args.corpus, "graph profile corpus")
            )
            _write_json(corpus.audit(), None)
            return 0
        if args.command == "plan-vllm-robust-graph-experiment":
            corpus = GraphProfileCorpus.from_dict(
                _load_object(args.corpus, "graph profile corpus")
            )
            promotion = promote_trace_preserving_graph_policy(
                corpus,
                minimum_train_regimes=args.minimum_train_regimes,
                minimum_holdout_regimes=args.minimum_holdout_regimes,
            )
            _write_json(promotion.artifact_dict(), args.promotion_output)
            if not promotion.eligible:
                sys.stderr.write(
                    "graph policy promotion rejected: "
                    + ", ".join(promotion.rejection_reasons)
                    + "\n"
                )
                return 2
            plan = build_promoted_graph_experiment_plan(
                corpus,
                promotion,
                campaign_id=args.campaign_id,
                blocks=args.blocks,
                pair_seeds=args.pair_seeds,
                include_replay_control=not args.exclude_replay_control,
            )
            _write_json(plan.artifact_dict(), args.output)
            return 0
        if args.command == "plan-vllm-coverage-graph-experiment":
            corpus = GraphProfileCorpus.from_dict(
                _load_object(args.corpus, "graph profile corpus")
            )
            train_constraints = GraphBucketConstraints(
                minimum_retained_graph_fraction=(args.minimum_retained_graph_fraction),
                maximum_remapped_graph_fraction=(args.maximum_remapped_graph_fraction),
                maximum_added_padding_ratio=args.maximum_added_padding_ratio,
            )
            search = search_coverage_constrained_graph_policy(
                corpus,
                policy_id=args.policy_id,
                constraints=train_constraints,
                holdout_constraints=GraphBucketConstraints(
                    minimum_retained_graph_fraction=(
                        args.holdout_minimum_retained_graph_fraction
                        if args.holdout_minimum_retained_graph_fraction is not None
                        else train_constraints.minimum_retained_graph_fraction
                    ),
                    maximum_remapped_graph_fraction=(
                        args.holdout_maximum_remapped_graph_fraction
                        if args.holdout_maximum_remapped_graph_fraction is not None
                        else train_constraints.maximum_remapped_graph_fraction
                    ),
                    maximum_added_padding_ratio=(
                        args.holdout_maximum_added_padding_ratio
                        if args.holdout_maximum_added_padding_ratio is not None
                        else train_constraints.maximum_added_padding_ratio
                    ),
                ),
                candidate_engine_roles=args.candidate_engine_role,
                minimum_train_regimes=args.minimum_train_regimes,
                minimum_holdout_regimes=args.minimum_holdout_regimes,
            )
            _write_json(search.artifact_dict(), args.search_output)
            if not search.eligible:
                sys.stderr.write(
                    "graph bucket policy rejected: "
                    + ", ".join(search.rejection_reasons)
                    + "\n"
                )
                return 2
            plan = build_coverage_constrained_graph_experiment_plan(
                corpus,
                search,
                campaign_id=args.campaign_id,
                blocks=args.blocks,
                pair_seeds=args.pair_seeds,
                include_replay_control=not args.exclude_replay_control,
            )
            _write_json(plan.artifact_dict(), args.output)
            return 0
        if args.command == "plan-vllm-graph-experiment":
            profile = VLLMGraphMetricsProfile.from_dict(
                _load_object(args.profile, "vLLM graph metrics profile")
            )
            plan = build_graph_experiment_plan(
                profile,
                campaign_id=args.campaign_id,
                blocks=args.blocks,
                pair_seeds=args.pair_seeds,
                include_replay_control=args.include_replay_control,
            )
            _write_json(plan.artifact_dict(), args.output)
            if args.output is not None:
                sys.stdout.write(json.dumps(plan.audit(), sort_keys=True) + "\n")
            return 0
        if args.command == "audit-vllm-graph-experiment":
            plan = GraphExperimentPlan.from_dict(
                _load_object(args.plan, "graph experiment plan")
            )
            _write_json(plan.audit(), None)
            return 0
        if args.command == "build-vllm-graph-calibration":
            template = CalibrationSpec.from_dict(
                _load_object(args.template, "calibration template")
            )
            graph_plan = GraphExperimentPlan.from_dict(
                _load_object(args.graph_plan, "graph experiment plan")
            )
            spec = build_graph_policy_calibration_spec(
                template,
                graph_plan,
                campaign_id=args.campaign_id,
                candidate_policy_ids=args.candidate_policy_id,
                include_replay_control=args.include_replay_control,
            )
            _write_json(spec.to_dict(), args.output)
            return 0
        if args.command == "evaluate-vllm-graph-policy":
            graph_plan = GraphExperimentPlan.from_dict(
                _load_object(args.graph_plan, "graph experiment plan")
            )
            profile = VLLMGraphMetricsProfile.from_dict(
                _load_object(args.profile, "vLLM graph metrics profile")
            )
            policy = next(
                (
                    policy
                    for policy in graph_plan.policies
                    if policy.policy_id == args.policy_id
                ),
                None,
            )
            if policy is None:
                raise ValueError(f"unknown graph policy: {args.policy_id}")
            coverage = evaluate_graph_policy_coverage(policy, profile)
            _write_json(coverage.artifact_dict(), args.output)
            return 0 if coverage.mapping_exact else 2
        if args.command == "assess-vllm-graph-experiment":
            plan = GraphExperimentPlan.from_dict(
                _load_object(args.plan, "graph experiment plan")
            )
            assessment = assess_graph_experiment(
                plan,
                args.results,
                comparison_id=args.comparison_id,
                minimum_requests_per_run=args.minimum_requests,
                minimum_blocks=args.minimum_blocks,
            )
            _write_json(assessment.artifact_dict(), args.output)
            return 0 if assessment.formal_claim_eligible else 2
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"inference-autopilot: error: {error}\n")
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
