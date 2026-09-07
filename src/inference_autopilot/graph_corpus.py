"""Regime-labelled graph profiles and fail-closed policy promotion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from inference_autopilot.calibration.models import (
    canonical_sha256,
    expect_keys,
    require_digest,
    require_id,
    require_object,
)
from inference_autopilot.graph_experiment import (
    GraphExperimentPlan,
    GraphPolicy,
    GraphPolicyCoverage,
    build_graph_experiment_plan,
    evaluate_graph_policy_coverage,
)
from inference_autopilot.vllm_graph_metrics import (
    VLLMGraphMetricsProfile,
    merge_vllm_graph_metrics,
)


@dataclass(frozen=True, slots=True)
class GraphProfileCorpusMember:
    regime_id: str
    split: str
    profile: VLLMGraphMetricsProfile

    def __post_init__(self) -> None:
        require_id(self.regime_id, "graph corpus regime_id")
        if self.split not in {"train", "holdout"}:
            raise ValueError("graph corpus split must be train or holdout")

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime_id": self.regime_id,
            "split": self.split,
            "profile": self.profile.artifact_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphProfileCorpusMember":
        expect_keys(raw, {"regime_id", "split", "profile"}, "graph corpus member")
        return cls(
            regime_id=str(raw["regime_id"]),
            split=str(raw["split"]),
            profile=VLLMGraphMetricsProfile.from_dict(
                require_object(raw["profile"], "graph corpus member profile")
            ),
        )


@dataclass(frozen=True, slots=True)
class GraphProfileCorpus:
    corpus_id: str
    members: tuple[GraphProfileCorpusMember, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported graph profile corpus: {self.schema_version}")
        require_id(self.corpus_id, "graph profile corpus_id")
        if len(self.members) < 2:
            raise ValueError("graph profile corpus requires at least two profiles")
        profile_ids = [member.profile.profile_id for member in self.members]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("graph profile corpus profile ids must be unique")
        profile_digests = [
            canonical_sha256(member.profile.to_dict()) for member in self.members
        ]
        if len(profile_digests) != len(set(profile_digests)):
            raise ValueError("graph profile corpus cannot contain duplicate profiles")
        source_digests = [member.profile.source_log_sha256 for member in self.members]
        if len(source_digests) != len(set(source_digests)):
            raise ValueError("graph profile corpus cannot reuse the same source log")
        regime_splits: dict[str, str] = {}
        for member in self.members:
            previous = regime_splits.setdefault(member.regime_id, member.split)
            if previous != member.split:
                raise ValueError(
                    f"graph corpus regime {member.regime_id} leaks across splits"
                )
        if {member.split for member in self.members} != {"train", "holdout"}:
            raise ValueError("graph profile corpus requires train and holdout profiles")
        reference = _engine_signature(self.members[0].profile)
        for member in self.members[1:]:
            if _engine_signature(member.profile) != reference:
                raise ValueError(
                    "graph corpus profiles must use identical roles, modes and "
                    "configured capture sizes"
                )

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "corpus_id": self.corpus_id,
            "members": [
                member.to_dict()
                for member in sorted(
                    self.members,
                    key=lambda item: (
                        item.split,
                        item.regime_id,
                        item.profile.profile_id,
                    ),
                )
            ],
        }

    def artifact_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {**payload, "graph_profile_corpus_sha256": canonical_sha256(payload)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphProfileCorpus":
        expect_keys(
            raw,
            {
                "schema_version",
                "corpus_id",
                "members",
                "graph_profile_corpus_sha256",
            },
            "graph profile corpus",
        )
        payload = dict(raw)
        digest = str(payload.pop("graph_profile_corpus_sha256"))
        require_digest(digest, "graph profile corpus SHA256")
        if canonical_sha256(payload) != digest:
            raise ValueError("graph profile corpus SHA256 does not match its content")
        members = payload["members"]
        if not isinstance(members, list):
            raise ValueError("graph profile corpus members must be an array")
        return cls(
            schema_version=str(payload["schema_version"]),
            corpus_id=str(payload["corpus_id"]),
            members=tuple(
                GraphProfileCorpusMember.from_dict(
                    require_object(item, "graph corpus member")
                )
                for item in members
            ),
        )

    def audit(self) -> dict[str, Any]:
        splits: dict[str, dict[str, Any]] = {}
        for split in ("train", "holdout"):
            members = [member for member in self.members if member.split == split]
            splits[split] = {
                "profile_count": len(members),
                "regime_count": len({member.regime_id for member in members}),
                "regime_ids": sorted({member.regime_id for member in members}),
                "profile_ids": sorted(member.profile.profile_id for member in members),
            }
        return {
            "corpus_id": self.corpus_id,
            "graph_profile_corpus_sha256": self.digest,
            "splits": splits,
        }


@dataclass(frozen=True, slots=True)
class GraphPolicyPromotion:
    corpus_id: str
    corpus_sha256: str
    policy: GraphPolicy
    minimum_train_regimes: int
    minimum_holdout_regimes: int
    train_regime_count: int
    holdout_regime_count: int
    eligible: bool
    rejection_reasons: tuple[str, ...]
    coverage: tuple[Mapping[str, Any], ...]
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "corpus_id": self.corpus_id,
            "corpus_sha256": self.corpus_sha256,
            "policy": self.policy.to_dict(),
            "minimum_train_regimes": self.minimum_train_regimes,
            "minimum_holdout_regimes": self.minimum_holdout_regimes,
            "train_regime_count": self.train_regime_count,
            "holdout_regime_count": self.holdout_regime_count,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
            "coverage": [dict(item) for item in self.coverage],
        }

    def artifact_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {**payload, "graph_policy_promotion_sha256": canonical_sha256(payload)}


def _engine_signature(
    profile: VLLMGraphMetricsProfile,
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    return tuple(
        (
            engine.engine_role,
            engine.graph_mode,
            engine.configured_capture_sizes,
        )
        for engine in sorted(profile.engines, key=lambda item: item.engine_role)
    )


def _positive_int(value: int, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def build_graph_profile_corpus(
    *,
    corpus_id: str,
    train_profiles: Sequence[tuple[str, VLLMGraphMetricsProfile]],
    holdout_profiles: Sequence[tuple[str, VLLMGraphMetricsProfile]],
) -> GraphProfileCorpus:
    members = tuple(
        GraphProfileCorpusMember(regime_id, "train", profile)
        for regime_id, profile in train_profiles
    ) + tuple(
        GraphProfileCorpusMember(regime_id, "holdout", profile)
        for regime_id, profile in holdout_profiles
    )
    return GraphProfileCorpus(corpus_id=corpus_id, members=members)


def promote_trace_preserving_graph_policy(
    corpus: GraphProfileCorpus,
    *,
    minimum_train_regimes: int,
    minimum_holdout_regimes: int,
) -> GraphPolicyPromotion:
    """Derive on train profiles and reject policies that fail independent holdouts."""

    _positive_int(minimum_train_regimes, "minimum_train_regimes")
    _positive_int(minimum_holdout_regimes, "minimum_holdout_regimes")
    train_members = tuple(
        member for member in corpus.members if member.split == "train"
    )
    holdout_members = tuple(
        member for member in corpus.members if member.split == "holdout"
    )
    train_regime_count = len({member.regime_id for member in train_members})
    holdout_regime_count = len({member.regime_id for member in holdout_members})
    if len(train_members) == 1:
        training_profile = train_members[0].profile
    else:
        training_profile = merge_vllm_graph_metrics(
            [member.profile for member in train_members],
            profile_id=f"{corpus.corpus_id}-training",
        )
    draft = build_graph_experiment_plan(
        training_profile,
        campaign_id=f"{corpus.corpus_id}-promotion-draft",
        blocks=1,
        pair_seeds=(0, 1),
    )
    policy = next(
        (
            item
            for item in draft.policies
            if item.policy_id == "trace-preserving-pruned"
        ),
        next(item for item in draft.policies if item.policy_id == "vllm-default"),
    )
    default = next(item for item in draft.policies if item.policy_id == "vllm-default")

    coverage_rows: list[Mapping[str, Any]] = []
    coverage_objects: list[GraphPolicyCoverage] = []
    for member in sorted(
        corpus.members,
        key=lambda item: (item.split, item.regime_id, item.profile.profile_id),
    ):
        result = evaluate_graph_policy_coverage(policy, member.profile)
        coverage_objects.append(result)
        coverage_rows.append(
            {
                "regime_id": member.regime_id,
                "split": member.split,
                "profile_id": member.profile.profile_id,
                "coverage": result.artifact_dict(),
            }
        )

    reasons = []
    if train_regime_count < minimum_train_regimes:
        reasons.append("insufficient_train_regimes")
    if holdout_regime_count < minimum_holdout_regimes:
        reasons.append("insufficient_holdout_regimes")
    if policy.engines == default.engines:
        reasons.append("no_policy_change")
    if any(
        not result.mapping_exact
        for member, result in zip(
            sorted(
                corpus.members,
                key=lambda item: (
                    item.split,
                    item.regime_id,
                    item.profile.profile_id,
                ),
            ),
            coverage_objects,
            strict=True,
        )
        if member.split == "holdout"
    ):
        reasons.append("holdout_mapping_not_exact")
    return GraphPolicyPromotion(
        corpus_id=corpus.corpus_id,
        corpus_sha256=corpus.digest,
        policy=policy,
        minimum_train_regimes=minimum_train_regimes,
        minimum_holdout_regimes=minimum_holdout_regimes,
        train_regime_count=train_regime_count,
        holdout_regime_count=holdout_regime_count,
        eligible=not reasons,
        rejection_reasons=tuple(reasons),
        coverage=tuple(coverage_rows),
    )


def build_promoted_graph_experiment_plan(
    corpus: GraphProfileCorpus,
    promotion: GraphPolicyPromotion,
    *,
    campaign_id: str,
    blocks: int,
    pair_seeds: Sequence[int],
    include_replay_control: bool = True,
) -> GraphExperimentPlan:
    if promotion.corpus_sha256 != corpus.digest:
        raise ValueError("graph policy promotion belongs to a different corpus")
    if not promotion.eligible:
        raise ValueError(
            "graph policy is not eligible: " + ", ".join(promotion.rejection_reasons)
        )
    train_profiles = [
        member.profile for member in corpus.members if member.split == "train"
    ]
    if len(train_profiles) == 1:
        training_profile = train_profiles[0]
    else:
        training_profile = merge_vllm_graph_metrics(
            train_profiles,
            profile_id=f"{corpus.corpus_id}-training",
        )
    plan = build_graph_experiment_plan(
        training_profile,
        campaign_id=campaign_id,
        blocks=blocks,
        pair_seeds=pair_seeds,
        include_replay_control=include_replay_control,
    )
    selected = next(
        (
            policy
            for policy in plan.policies
            if policy.policy_id == promotion.policy.policy_id
        ),
        None,
    )
    if selected != promotion.policy:
        raise ValueError("promoted graph policy changed while rebuilding experiment")
    return replace(plan, source_profile_sha256=corpus.digest)
