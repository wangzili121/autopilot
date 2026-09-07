"""Inference Graph IR adapters for chang's Conditional IS paths."""

from __future__ import annotations

from inference_autopilot.ir import InferenceGraph, IntExpr, LoopSpec, ParameterSpec, StageSpec


def _build_conditional_is_graph(
    *,
    candidate_count: int = 4,
    rollout_count: int = 4,
    block_size: int = 16,
    total_length: int = 128,
    apply_importance_correction: bool = True,
    small_proposal: bool,
) -> InferenceGraph:
    """Build a concrete graph while preserving structured cardinality expressions."""

    if block_size > total_length:
        raise ValueError("block_size cannot exceed total_length")

    parameters = (
        ParameterSpec(
            "candidate_count",
            "integer",
            candidate_count,
            "algorithm_budget",
            "Base-model candidate blocks sampled per generation step.",
            minimum=1,
            tunable=True,
            quality_sensitive=True,
        ),
        ParameterSpec(
            "rollout_count",
            "integer",
            rollout_count,
            "algorithm_budget",
            (
                "Proposal-model continuations evaluated per non-terminal candidate."
                if small_proposal
                else "Base-model continuations evaluated per non-terminal candidate."
            ),
            minimum=1,
            tunable=True,
            quality_sensitive=True,
        ),
        ParameterSpec(
            "block_size",
            "integer",
            block_size,
            "algorithm_budget",
            "Maximum candidate tokens committed by one generation step.",
            minimum=1,
            tunable=True,
            quality_sensitive=True,
        ),
        ParameterSpec(
            "total_length",
            "integer",
            total_length,
            "algorithm_budget",
            "Maximum generated sequence length.",
            minimum=1,
            tunable=False,
            quality_sensitive=True,
        ),
        ParameterSpec(
            "apply_importance_correction",
            "boolean",
            apply_importance_correction,
            "algorithm_semantics",
            (
                "Whether the base model scores off-policy rollouts for p/q correction."
                if small_proposal
                else (
                    "On-policy rollouts reuse generation log-probabilities; "
                    "the p/q ratio is one."
                )
            ),
            tunable=False,
            quality_sensitive=True,
        ),
        ParameterSpec(
            "generated_length_before",
            "integer",
            0,
            "workload_state",
            "Generated tokens already committed before the current step.",
            minimum=0,
        ),
    )

    candidates = IntExpr.symbol("candidate_count")
    rollouts = IntExpr.compound(
        "multiply", candidates, IntExpr.symbol("rollout_count")
    )
    remaining = IntExpr.compound(
        "subtract",
        IntExpr.symbol("total_length"),
        IntExpr.symbol("generated_length_before"),
    )
    candidate_tokens = IntExpr.compound(
        "minimum", IntExpr.symbol("block_size"), remaining
    )
    rollout_tokens = IntExpr.compound(
        "maximum",
        IntExpr.literal(0),
        IntExpr.compound("subtract", remaining, candidate_tokens),
    )

    rollout_stage_id = (
        "proposal_rollout_generate" if small_proposal else "rollout_generate"
    )
    rollout_runtime_fields = (
        (
            "runtime.model_runner",
            "runtime.tensor_parallel_size",
            "runtime.data_parallel_size",
            "runtime.pipeline_parallel_size",
            "proposal.max_num_seqs",
            "proposal.max_num_batched_tokens",
            "proposal.memory_fraction",
            "proposal.batch_wait_seconds",
            "proposal.graph_mode",
            "proposal.capture_sizes",
            "proposal.stage_wavefront_mode",
            "proposal.stage_wavefront_min_utilization",
            "proposal.stage_wavefront_max_wait_seconds",
        )
        if small_proposal
        else (
            "runtime.model_runner",
            "runtime.tensor_parallel_size",
            "runtime.data_parallel_size",
            "runtime.pipeline_parallel_size",
            "base.max_num_seqs",
            "base.max_num_batched_tokens",
            "base.memory_fraction",
            "base.batch_wait_seconds",
            "base.graph_mode",
            "base.capture_sizes",
        )
    )

    stages = [
        StageSpec(
            stage_id="candidate_generate",
            primitive="generate",
            resource_role="base_model",
            multiplicity_upper_bound=candidates,
            token_extent_upper_bound=candidate_tokens,
            cache_key_parts=("prompt", "generated_prefix"),
            tunable_runtime_fields=(
                "runtime.model_runner",
                "runtime.tensor_parallel_size",
                "runtime.data_parallel_size",
                "runtime.pipeline_parallel_size",
                "base.max_num_seqs",
                "base.max_num_batched_tokens",
                "base.memory_fraction",
                "base.batch_wait_seconds",
                "base.graph_mode",
                "base.capture_sizes",
            ),
            description="Sample candidate blocks with the base policy.",
        ),
        StageSpec(
            stage_id=rollout_stage_id,
            primitive="generate",
            resource_role="proposal_model" if small_proposal else "base_model",
            multiplicity_upper_bound=rollouts,
            token_extent_upper_bound=rollout_tokens,
            depends_on=("candidate_generate",),
            cache_key_parts=("prompt", "generated_prefix", "candidate_tokens"),
            tunable_runtime_fields=rollout_runtime_fields,
            description=(
                "Generate off-policy rollout suffixes; EOS may reduce actual work."
                if small_proposal
                else (
                    "Generate on-policy rollout suffixes on the same model; "
                    "EOS may reduce actual work."
                )
            ),
        ),
    ]

    reduction_dependencies = ["reward_evaluate"]
    if small_proposal and apply_importance_correction:
        stages.append(
            StageSpec(
                stage_id="target_score",
                primitive="score",
                resource_role="base_model",
                multiplicity_upper_bound=rollouts,
                token_extent_upper_bound=rollout_tokens,
                depends_on=(rollout_stage_id,),
                cache_key_parts=(
                    "prompt",
                    "generated_prefix",
                    "candidate_tokens",
                    "rollout_tokens",
                    "base_policy",
                ),
                tunable_runtime_fields=(
                    "runtime.model_runner",
                    "base.max_num_seqs",
                    "base.max_num_batched_tokens",
                    "base.memory_fraction",
                    "base.score_priority",
                    "base.graph_mode",
                    "base.capture_sizes",
                ),
                description="Teacher-force rollout tokens under the base policy for p/q.",
            )
        )
        reduction_dependencies.append("target_score")

    stages.extend(
        (
            StageSpec(
                stage_id="reward_evaluate",
                primitive="reward",
                resource_role="cpu",
                multiplicity_upper_bound=rollouts,
                token_extent_upper_bound=None,
                depends_on=(rollout_stage_id,),
                batching_scope="host_vectorized",
                description="Evaluate completed trajectories with the configured reward.",
            ),
            StageSpec(
                stage_id="importance_reduce",
                primitive="reduce",
                resource_role="cpu",
                multiplicity_upper_bound=candidates,
                token_extent_upper_bound=None,
                depends_on=tuple(reduction_dependencies),
                batching_scope="host_vectorized",
                description="Compute rollout weights and log-mean-exp per candidate.",
            ),
            StageSpec(
                stage_id="candidate_select",
                primitive="select",
                resource_role="cpu",
                multiplicity_upper_bound=IntExpr.literal(1),
                token_extent_upper_bound=None,
                depends_on=("importance_reduce",),
                batching_scope="single_request",
                description="Sample one candidate from normalized conditional weights.",
            ),
        )
    )

    loop = LoopSpec(
        stage_ids=tuple(stage.stage_id for stage in stages),
        maximum_iterations=IntExpr.compound(
            "ceiling_divide",
            IntExpr.symbol("total_length"),
            IntExpr.symbol("block_size"),
        ),
        state_variable="generated_length_before",
        state_update="append selected candidate block and increase generated_length_before",
        early_exit_conditions=("selected candidate contains eos", "total_length reached"),
    )
    return InferenceGraph(
        algorithm_id=(
            "conditional_is_small_proposal" if small_proposal else "conditional_is"
        ),
        algorithm_semantics=(
            "exact"
            if not small_proposal or apply_importance_correction
            else "biased_ablation"
        ),
        parameters=parameters,
        stages=tuple(stages),
        loop=loop,
        source_contract=(
            "inference_scaling.arllm.algorithms.conditional_is._sample_candidates",
            "inference_scaling.arllm.algorithms.conditional_is.estimate_conditional_weights",
            "inference_scaling.arllm.algorithms.conditional_is.AutoregressiveStepwiseAdapter",
        ),
        metadata={
            "cardinality_kind": "upper_bound_due_to_terminal_candidates_and_eos",
            "dependency_semantics": "data_readiness_not_physical_stage_batching",
            "runtime_owner": (
                "algorithm adapter submits work; serving engines own continuous batching"
            ),
            "execution_path": (
                "small_proposal_off_policy"
                if small_proposal
                else "same_model_on_policy"
            ),
            "generation_logprobs_reused": not small_proposal,
            "python_stage_barriers": (
                ["candidate_generate_to_rollout_generate", "rollout_generate_to_reward"]
                if not small_proposal
                else [
                    "candidate_generate_to_proposal_rollout_generate",
                    "proposal_rollout_generate_to_target_score_and_reward",
                ]
            ),
        },
    )


def build_conditional_is_graph(
    *,
    candidate_count: int = 4,
    rollout_count: int = 4,
    block_size: int = 16,
    total_length: int = 128,
) -> InferenceGraph:
    """Describe normal Conditional IS with on-policy rollouts on one model."""

    return _build_conditional_is_graph(
        candidate_count=candidate_count,
        rollout_count=rollout_count,
        block_size=block_size,
        total_length=total_length,
        apply_importance_correction=True,
        small_proposal=False,
    )


def build_conditional_is_small_proposal_graph(
    *,
    candidate_count: int = 4,
    rollout_count: int = 4,
    block_size: int = 16,
    total_length: int = 128,
    apply_importance_correction: bool = True,
) -> InferenceGraph:
    """Describe Conditional IS with off-policy rollouts from a smaller model."""

    return _build_conditional_is_graph(
        candidate_count=candidate_count,
        rollout_count=rollout_count,
        block_size=block_size,
        total_length=total_length,
        apply_importance_correction=apply_importance_correction,
        small_proposal=True,
    )
