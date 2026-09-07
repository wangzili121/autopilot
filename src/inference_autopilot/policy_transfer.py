"""Fail-closed transfer experiments for a policy outside its fitted regime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from math import isfinite
from typing import Any

from inference_autopilot.calibration.models import (
    CalibrationSpec,
    ConfigurationSpec,
    ProtocolSpec,
    WorkloadContract,
    canonical_json,
    canonical_sha256,
    expect_keys,
    require_digest,
    require_id,
    require_object,
)
from inference_autopilot.features import deployment_features
from inference_autopilot.policy_selection import PolicyBundle
from inference_autopilot.runners.chang import arrival_trace_sha256


_PLAN_STATUSES = {"planned", "blocked"}
_PLAN_KEYS = {
    "schema_version",
    "producer",
    "transfer_id",
    "status",
    "source",
    "source_context",
    "target_context",
    "policy",
    "guard_evaluation",
    "calibration_spec",
    "audit",
    "policy_transfer_plan_sha256",
}


def _same_value(left: Any, right: Any) -> bool:
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return float(left) == float(right)
    return canonical_json({"value": left}) == canonical_json({"value": right})


def _configuration_matches(
    complete: Mapping[str, Any], required: Mapping[str, Any]
) -> bool:
    return all(
        name in complete and _same_value(complete[name], value)
        for name, value in required.items()
    )


@dataclass(frozen=True, slots=True)
class PolicyTransferSpec:
    transfer_id: str
    policy_bundle_sha256: str
    campaign_id: str
    protocol: ProtocolSpec
    target_workload_contract: WorkloadContract
    allowed_guard_deviations: tuple[str, ...]
    include_replay_control: bool = True
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported policy transfer spec: {self.schema_version}")
        require_id(self.transfer_id, "policy transfer_id")
        require_id(self.campaign_id, "policy transfer campaign_id")
        require_digest(self.policy_bundle_sha256, "policy_bundle_sha256")
        if not isinstance(self.include_replay_control, bool):
            raise ValueError("include_replay_control must be boolean")
        if tuple(
            sorted(set(self.allowed_guard_deviations))
        ) != self.allowed_guard_deviations or any(
            not item for item in self.allowed_guard_deviations
        ):
            raise ValueError("allowed guard deviations must be sorted and unique")
        expected_arrival = arrival_trace_sha256(
            self.target_workload_contract.parameters
        )
        if expected_arrival != self.target_workload_contract.arrival_trace_sha256:
            raise ValueError(
                "target workload arrival trace SHA256 does not match parameters"
            )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "transfer_id": self.transfer_id,
            "policy_bundle_sha256": self.policy_bundle_sha256,
            "campaign_id": self.campaign_id,
            "protocol": self.protocol.to_dict(),
            "target_workload_contract": self.target_workload_contract.to_dict(),
            "allowed_guard_deviations": list(self.allowed_guard_deviations),
            "include_replay_control": self.include_replay_control,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyTransferSpec":
        keys = {
            "schema_version",
            "transfer_id",
            "policy_bundle_sha256",
            "campaign_id",
            "protocol",
            "target_workload_contract",
            "allowed_guard_deviations",
            "include_replay_control",
        }
        expect_keys(raw, keys, "policy transfer spec")
        allowed = raw["allowed_guard_deviations"]
        if not isinstance(allowed, list) or any(
            not isinstance(item, str) for item in allowed
        ):
            raise ValueError("allowed_guard_deviations must be strings")
        if not isinstance(raw["include_replay_control"], bool):
            raise ValueError("include_replay_control must be boolean")
        return cls(
            schema_version=str(raw["schema_version"]),
            transfer_id=str(raw["transfer_id"]),
            policy_bundle_sha256=str(raw["policy_bundle_sha256"]),
            campaign_id=str(raw["campaign_id"]),
            protocol=ProtocolSpec.from_dict(
                require_object(raw["protocol"], "policy transfer protocol")
            ),
            target_workload_contract=WorkloadContract.from_dict(
                require_object(
                    raw["target_workload_contract"], "target workload contract"
                )
            ),
            allowed_guard_deviations=tuple(allowed),
            include_replay_control=raw["include_replay_control"],
        )


@dataclass(frozen=True, slots=True)
class PolicyTransferPlan:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        expect_keys(self.payload, _PLAN_KEYS, "policy transfer plan")
        raw = dict(self.payload)
        digest = str(raw.pop("policy_transfer_plan_sha256", ""))
        require_digest(digest, "policy_transfer_plan_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("policy transfer plan SHA256 does not match its content")
        if raw["schema_version"] != "1.0":
            raise ValueError(
                f"unsupported policy transfer plan: {raw['schema_version']}"
            )
        if raw["status"] not in _PLAN_STATUSES:
            raise ValueError(f"unsupported policy transfer status: {raw['status']}")
        require_id(str(raw["transfer_id"]), "policy transfer_id")
        source = require_object(raw["source"], "policy transfer source")
        expect_keys(
            source,
            {
                "transfer_spec_sha256",
                "policy_bundle_sha256",
                "calibration_template_sha256",
            },
            "policy transfer source",
        )
        for name, value in source.items():
            require_digest(str(value), name)
        for name in (
            "source_context",
            "target_context",
            "policy",
            "guard_evaluation",
            "audit",
        ):
            require_object(raw[name], f"policy transfer {name}")
        guard = raw["guard_evaluation"]
        for name in (
            "allowed_deviations",
            "deviations",
            "unallowed_deviations",
            "deferred_runtime_checks",
            "within_range_checks",
        ):
            if not isinstance(guard.get(name), list):
                raise ValueError(f"policy transfer guard {name} must be an array")
        calibration = raw["calibration_spec"]
        if raw["status"] == "planned":
            if not isinstance(calibration, Mapping):
                raise ValueError("planned policy transfer requires a calibration spec")
            CalibrationSpec.from_dict(calibration)
            if guard["unallowed_deviations"]:
                raise ValueError(
                    "planned policy transfer has unallowed guard deviations"
                )
        else:
            if calibration is not None:
                raise ValueError(
                    "blocked policy transfer cannot contain a calibration spec"
                )
            if not guard["unallowed_deviations"]:
                raise ValueError("blocked policy transfer requires a guard violation")
        canonical = canonical_json(self.payload)
        object.__setattr__(self, "payload", json.loads(canonical))

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.payload))

    def audit(self) -> dict[str, Any]:
        return {
            "transfer_id": self.payload["transfer_id"],
            "status": self.status,
            **dict(self.payload["audit"]),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyTransferPlan":
        return cls(raw)


def _guard_evaluation(
    spec: PolicyTransferSpec,
    policy: PolicyBundle,
    template: CalibrationSpec,
    selected_settings: Mapping[str, Any],
) -> dict[str, Any]:
    guard = require_object(policy.payload["activation_guard"], "activation guard")
    target_parameters = spec.target_workload_contract.parameters
    observed_features = deployment_features(selected_settings)
    observed_features.update(
        {
            f"workload.{name}": value
            for name, value in target_parameters.items()
            if isinstance(value, (str, int, float, bool))
            and not (isinstance(value, float) and not isfinite(value))
        }
    )
    deviations: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    within_ranges: list[dict[str, Any]] = []

    def add_deviation(code: str, field: str, expected: Any, actual: Any) -> None:
        deviations.append(
            {
                "code": code,
                "field": field,
                "expected": expected,
                "actual": actual,
                "allowed": code in spec.allowed_guard_deviations,
            }
        )

    if guard["environment_id"] != template.environment_contract.environment_id:
        add_deviation(
            "environment_id",
            "environment_id",
            guard["environment_id"],
            template.environment_contract.environment_id,
        )
    if guard["workload_id"] != spec.target_workload_contract.workload_id:
        add_deviation(
            "workload_id",
            "workload_id",
            guard["workload_id"],
            spec.target_workload_contract.workload_id,
        )

    for name, expected in sorted(guard["exact_static_features"].items()):
        if name not in observed_features:
            if name.startswith("deployment."):
                raise ValueError(
                    f"selected policy omits guarded deployment feature {name}"
                )
            deferred.append(
                {
                    "field": name,
                    "expected": expected,
                    "reason": "feature_is_observed_only_after_workload_materialization",
                }
            )
        elif not _same_value(expected, observed_features[name]):
            add_deviation(
                f"exact_static_feature:{name}",
                name,
                expected,
                observed_features[name],
            )

    for name, bounds in sorted(guard["static_feature_ranges"].items()):
        if name not in observed_features:
            deferred.append(
                {
                    "field": name,
                    "reason": "range_must_be_checked_from_runtime_workload_features",
                    "minimum": bounds["minimum"],
                    "maximum": bounds["maximum"],
                }
            )
            continue
        value = observed_features[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            add_deviation(f"static_feature_range:{name}", name, bounds, value)
            continue
        if not float(bounds["minimum"]) <= float(value) <= float(bounds["maximum"]):
            add_deviation(f"static_feature_range:{name}", name, bounds, value)
        else:
            within_ranges.append(
                {
                    "field": name,
                    "minimum": bounds["minimum"],
                    "maximum": bounds["maximum"],
                    "actual": value,
                }
            )

    deviations.sort(key=lambda item: item["code"])
    deferred.sort(key=lambda item: item["field"])
    within_ranges.sort(key=lambda item: item["field"])
    return {
        "on_violation": guard["on_violation"],
        "allowed_deviations": list(spec.allowed_guard_deviations),
        "deviations": deviations,
        "unallowed_deviations": [
            item["code"] for item in deviations if not item["allowed"]
        ],
        "deferred_runtime_checks": deferred,
        "within_range_checks": within_ranges,
    }


def plan_policy_transfer(
    spec: PolicyTransferSpec,
    policy: PolicyBundle,
    calibration_template: CalibrationSpec,
) -> PolicyTransferPlan:
    """Compile a selected/fallback policy pair for a declared target regime."""

    policy_payload = policy.payload
    policy_digest = str(policy_payload["policy_bundle_sha256"])
    if spec.policy_bundle_sha256 != policy_digest:
        raise ValueError("policy transfer spec does not match policy bundle")
    if policy.status != "selected" or not isinstance(
        policy_payload["selected"], Mapping
    ):
        raise ValueError("policy transfer requires a selected policy")
    if not calibration_template.strong_baseline:
        raise ValueError("policy transfer requires a strong calibration baseline")
    guard = policy_payload["activation_guard"]
    semantic = calibration_template.semantic_contract
    if guard["algorithm_id"] != semantic.algorithm_id:
        raise ValueError("policy and calibration template algorithms do not match")
    if guard["graph_sha256"] != semantic.graph_sha256:
        raise ValueError("policy and calibration template graphs do not match")

    selected = require_object(policy_payload["selected"], "selected policy")
    fallback = require_object(policy_payload["fallback"], "fallback policy")
    selected_partial = require_object(
        selected["deployment_settings"], "selected deployment settings"
    )
    fallback_partial = require_object(
        fallback["deployment_settings"], "fallback deployment settings"
    )
    baseline_settings = dict(calibration_template.baseline.settings)
    if not _configuration_matches(baseline_settings, fallback_partial):
        raise ValueError("calibration template baseline does not match policy fallback")
    selected_settings = dict(baseline_settings)
    selected_settings.update(selected_partial)
    evaluation = _guard_evaluation(
        spec, policy, calibration_template, selected_settings
    )
    status = "blocked" if evaluation["unallowed_deviations"] else "planned"

    calibration: CalibrationSpec | None = None
    if status == "planned":
        baseline = ConfigurationSpec(
            configuration_id=f"{spec.campaign_id}-fallback",
            settings=baseline_settings,
            description=f"Strong fallback from policy {policy_payload['policy_id']}",
        )
        candidates = []
        if spec.include_replay_control:
            candidates.append(
                ConfigurationSpec(
                    configuration_id=f"{spec.campaign_id}-replay-control",
                    settings=baseline_settings,
                    description="Manifest-identical fallback replay control",
                )
            )
        candidates.append(
            ConfigurationSpec(
                configuration_id=f"{spec.campaign_id}-selected-policy",
                settings=selected_settings,
                description=(
                    f"Transfer candidate {selected['candidate_id']} from policy "
                    f"{policy_payload['policy_id']}"
                ),
            )
        )
        calibration = replace(
            calibration_template,
            campaign_id=spec.campaign_id,
            protocol=spec.protocol,
            workload_contract=spec.target_workload_contract,
            baseline=baseline,
            candidates=tuple(candidates),
        )

    source_context = {
        "workload_id": guard["workload_id"],
        "environment_id": guard["environment_id"],
        "algorithm_id": guard["algorithm_id"],
        "graph_sha256": guard["graph_sha256"],
    }
    target_context = {
        "workload_contract": spec.target_workload_contract.to_dict(),
        "environment_id": calibration_template.environment_contract.environment_id,
        "algorithm_id": semantic.algorithm_id,
        "graph_sha256": semantic.graph_sha256,
    }
    policy_summary = {
        "policy_id": policy_payload["policy_id"],
        "selected_candidate_id": selected["candidate_id"],
        "selected_deployment_settings": selected_settings,
        "fallback_candidate_id": fallback["candidate_id"],
        "fallback_deployment_settings": baseline_settings,
        "requires_engine_restart": bool(
            selected["requires_engine_restart"] or fallback["requires_engine_restart"]
        ),
    }
    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "transfer_id": spec.transfer_id,
        "status": status,
        "source": {
            "transfer_spec_sha256": spec.sha256,
            "policy_bundle_sha256": policy_digest,
            "calibration_template_sha256": canonical_sha256(
                calibration_template.to_dict()
            ),
        },
        "source_context": source_context,
        "target_context": target_context,
        "policy": policy_summary,
        "guard_evaluation": evaluation,
        "calibration_spec": None if calibration is None else calibration.to_dict(),
        "audit": {
            "guard_deviation_count": len(evaluation["deviations"]),
            "unallowed_guard_deviation_count": len(evaluation["unallowed_deviations"]),
            "deferred_runtime_check_count": len(evaluation["deferred_runtime_checks"]),
            "configuration_change_count": sum(
                1
                for name in set(baseline_settings) | set(selected_settings)
                if not _same_value(
                    baseline_settings.get(name), selected_settings.get(name)
                )
            ),
            "requires_engine_restart": policy_summary["requires_engine_restart"],
            "calibration_candidate_count": (
                0 if calibration is None else len(candidates)
            ),
        },
    }
    return PolicyTransferPlan(
        {**payload, "policy_transfer_plan_sha256": canonical_sha256(payload)}
    )


def calibration_spec_from_policy_transfer_plan(
    plan: PolicyTransferPlan,
) -> CalibrationSpec:
    if plan.status != "planned":
        raise ValueError("blocked policy transfer has no calibration spec")
    return CalibrationSpec.from_dict(plan.payload["calibration_spec"])
