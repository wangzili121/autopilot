"""Fail-closed execution protocol for state-isolated lifecycle plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from inference_autopilot.calibration.models import canonical_sha256, require_digest
from inference_autopilot.lifecycle_planning import EngineLifecyclePlan


_RECEIPT_KEYS = {
    "schema_version",
    "producer",
    "execution_mode",
    "status",
    "lifecycle_plan_sha256",
    "selected_strategy",
    "epochs",
    "summary",
    "lifecycle_execution_receipt_sha256",
}
_START_ACTIONS = {"start_engine", "start_resident_engine"}


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


@dataclass(frozen=True, slots=True)
class EpochMeasurement:
    native_result_sha256: str
    observation_sha256: str
    output_token_ids_sha256: str

    def __post_init__(self) -> None:
        require_digest(self.native_result_sha256, "native_result_sha256")
        require_digest(self.observation_sha256, "observation_sha256")
        require_digest(self.output_token_ids_sha256, "output_token_ids_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "native_result_sha256": self.native_result_sha256,
            "observation_sha256": self.observation_sha256,
            "output_token_ids_sha256": self.output_token_ids_sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EpochMeasurement":
        expected = {
            "native_result_sha256",
            "observation_sha256",
            "output_token_ids_sha256",
        }
        _expect_exact_keys(raw, expected, "epoch measurement")
        return cls(**{name: str(raw[name]) for name in expected})


class LifecycleDriver(Protocol):
    """Algorithm/backend-specific operations used by the generic state machine."""

    def start_engine(self, role: str, fingerprint: str) -> None: ...

    def stop_engine(self, role: str, fingerprint: str) -> None: ...

    def reset_epoch(
        self,
        run_id: str,
        workload_seed: int,
        requirements: Sequence[str],
        bindings: Mapping[str, str],
    ) -> Mapping[str, bool]: ...

    def measure_epoch(
        self,
        run_id: str,
        workload_seed: int,
        bindings: Mapping[str, str],
    ) -> EpochMeasurement: ...


def _receipt_summary(epochs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "epoch_count": len(epochs),
        "engine_start_count": sum(
            receipt["action"] in _START_ACTIONS
            for epoch in epochs
            for receipt in epoch["action_receipts"]
        ),
        "engine_stop_count": sum(
            receipt["action"] == "stop_engine"
            for epoch in epochs
            for receipt in epoch["action_receipts"]
        ),
        "reset_epoch_count": sum(
            any(
                receipt["action"] == "reset_epoch_state"
                for receipt in epoch["action_receipts"]
            )
            for epoch in epochs
        ),
        "measurement_count": sum("measurement" in epoch for epoch in epochs),
        "all_actions_acknowledged": all(
            receipt["status"] == "acknowledged"
            for epoch in epochs
            for receipt in epoch["action_receipts"]
        ),
        "all_resets_acknowledged": all(
            all(epoch["reset_acknowledgements"].values()) for epoch in epochs
        ),
        "formal_execution_eligible": False,
    }


@dataclass(frozen=True, slots=True)
class LifecycleExecutionReceipt:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _expect_exact_keys(self.payload, _RECEIPT_KEYS, "lifecycle execution receipt")
        raw = dict(self.payload)
        digest = str(raw.pop("lifecycle_execution_receipt_sha256", ""))
        require_digest(digest, "lifecycle_execution_receipt_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("lifecycle execution receipt SHA256 does not match content")
        if (
            raw["schema_version"] != "1.0"
            or raw["execution_mode"] != "validation_only"
            or raw["status"] != "diagnostic_complete"
        ):
            raise ValueError("unsupported lifecycle execution receipt")
        require_digest(str(raw["lifecycle_plan_sha256"]), "lifecycle_plan_sha256")
        epochs = raw["epochs"]
        if not isinstance(epochs, list) or not epochs:
            raise ValueError("lifecycle receipt epochs must be a non-empty list")
        for expected_index, epoch in enumerate(epochs):
            if not isinstance(epoch, Mapping):
                raise ValueError("lifecycle epoch receipt must be an object")
            _expect_exact_keys(
                epoch,
                {
                    "sequence_index",
                    "run_id",
                    "workload_seed",
                    "planned_actions_sha256",
                    "action_receipts",
                    "reset_acknowledgements",
                    "active_engine_fingerprints",
                    "measurement",
                },
                "lifecycle epoch receipt",
            )
            if epoch["sequence_index"] != expected_index:
                raise ValueError("lifecycle receipt sequence must be contiguous")
            if (
                not isinstance(epoch["run_id"], str)
                or not epoch["run_id"]
                or isinstance(epoch["workload_seed"], bool)
                or not isinstance(epoch["workload_seed"], int)
                or epoch["workload_seed"] < 0
            ):
                raise ValueError("lifecycle epoch identity is invalid")
            require_digest(str(epoch["planned_actions_sha256"]), "planned_actions_sha256")
            action_receipts = epoch["action_receipts"]
            if not isinstance(action_receipts, list) or not action_receipts:
                raise ValueError("lifecycle epoch requires action receipts")
            if any(
                not isinstance(receipt, Mapping)
                or receipt.get("status") != "acknowledged"
                or not isinstance(receipt.get("action"), str)
                for receipt in action_receipts
            ):
                raise ValueError("lifecycle action acknowledgement is invalid")
            acknowledgements = epoch["reset_acknowledgements"]
            if (
                not isinstance(acknowledgements, Mapping)
                or not acknowledgements
                or any(value is not True for value in acknowledgements.values())
            ):
                raise ValueError("lifecycle reset acknowledgement is invalid")
            bindings = epoch["active_engine_fingerprints"]
            if (
                not isinstance(bindings, Mapping)
                or set(bindings) != {"base", "proposal"}
            ):
                raise ValueError("lifecycle active engine bindings are invalid")
            for fingerprint in bindings.values():
                require_digest(str(fingerprint), "active engine fingerprint")
            EpochMeasurement.from_dict(epoch["measurement"])
        if raw["summary"] != _receipt_summary(epochs):
            raise ValueError("lifecycle execution summary does not match epoch receipts")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LifecycleExecutionReceipt":
        return cls(dict(raw))


def _fingerprint_roles(plan: Mapping[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for run in plan["engine_runs"]:
        for engine in run["engines"]:
            fingerprint = str(engine["engine_fingerprint_sha256"])
            role = str(engine["role"])
            previous = roles.setdefault(fingerprint, role)
            if previous != role:
                raise ValueError("one lifecycle fingerprint maps to multiple roles")
    return roles


def _expected_bindings(plan: Mapping[str, Any], sequence_index: int) -> dict[str, str]:
    run = plan["engine_runs"][sequence_index]
    return {
        str(engine["role"]): str(engine["engine_fingerprint_sha256"])
        for engine in run["engines"]
    }


def execute_lifecycle_plan(
    plan: EngineLifecyclePlan,
    driver: LifecycleDriver,
    *,
    validation_only: bool,
) -> LifecycleExecutionReceipt:
    """Execute a plan without granting its measurements formal evidence status."""

    if validation_only is not True:
        raise ValueError("lifecycle execution is available only in explicit validation mode")
    payload = plan.to_dict()
    if payload["status"] != "validation_required":
        raise ValueError("lifecycle plan is not awaiting validation")
    fingerprint_roles = _fingerprint_roles(payload)
    active: set[tuple[str, str]] = set()
    bindings: dict[str, str] = {}
    epochs: list[dict[str, Any]] = []

    def start(role: str, fingerprint: str) -> None:
        key = (role, fingerprint)
        if key in active:
            raise ValueError(f"engine is already active: {role}:{fingerprint}")
        driver.start_engine(role, fingerprint)
        active.add(key)

    def stop(role: str, fingerprint: str) -> None:
        key = (role, fingerprint)
        if key not in active:
            raise ValueError(f"engine is not active: {role}:{fingerprint}")
        driver.stop_engine(role, fingerprint)
        active.remove(key)
        if bindings.get(role) == fingerprint:
            bindings.pop(role)

    try:
        for expected_index, schedule in enumerate(payload["actions"]):
            if schedule["sequence_index"] != expected_index:
                raise ValueError("lifecycle action schedule is not contiguous")
            action_receipts = []
            reset_acknowledgements: dict[str, bool] | None = None
            for action in schedule["actions_before_measurement"]:
                kind = str(action["action"])
                fingerprint = str(action.get("fingerprint", ""))
                role = str(action.get("role", ""))
                if kind == "start_resident_engine":
                    role = fingerprint_roles[fingerprint]
                    start(role, fingerprint)
                elif kind == "start_engine":
                    start(role, fingerprint)
                    bindings[role] = fingerprint
                elif kind == "stop_engine":
                    stop(role, fingerprint)
                elif kind == "reuse_engine":
                    if (role, fingerprint) not in active or bindings.get(role) != fingerprint:
                        raise ValueError(f"cannot reuse inactive engine: {role}:{fingerprint}")
                elif kind == "bind_resident_engine":
                    if (role, fingerprint) not in active:
                        raise ValueError(f"cannot bind inactive engine: {role}:{fingerprint}")
                    bindings[role] = fingerprint
                elif kind == "reset_epoch_state":
                    if reset_acknowledgements is not None:
                        raise ValueError("an epoch contains more than one reset action")
                    requirements = tuple(str(value) for value in action["requirements"])
                    acknowledgements = driver.reset_epoch(
                        str(schedule["run_id"]),
                        int(schedule["workload_seed"]),
                        requirements,
                        dict(bindings),
                    )
                    reset_acknowledgements = {
                        str(name): value for name, value in acknowledgements.items()
                    }
                    if set(reset_acknowledgements) != set(requirements):
                        raise ValueError("driver reset acknowledgements do not match requirements")
                    if any(value is not True for value in reset_acknowledgements.values()):
                        raise ValueError("driver failed an epoch reset requirement")
                else:
                    raise ValueError(f"unsupported lifecycle action: {kind}")
                action_receipts.append({**action, "status": "acknowledged"})

            if reset_acknowledgements is None:
                raise ValueError("epoch measurement requires reset acknowledgement")
            expected_bindings = _expected_bindings(payload, expected_index)
            if bindings != expected_bindings:
                raise ValueError("active engine bindings differ from the planned run")
            measurement = driver.measure_epoch(
                str(schedule["run_id"]),
                int(schedule["workload_seed"]),
                dict(bindings),
            )
            for action in schedule["actions_after_measurement"]:
                kind = str(action["action"])
                if kind != "stop_engine":
                    raise ValueError(f"unsupported post-measurement action: {kind}")
                stop(str(action["role"]), str(action["fingerprint"]))
                action_receipts.append({**action, "status": "acknowledged"})
            epochs.append(
                {
                    "sequence_index": expected_index,
                    "run_id": schedule["run_id"],
                    "workload_seed": schedule["workload_seed"],
                    "planned_actions_sha256": canonical_sha256(schedule),
                    "action_receipts": action_receipts,
                    "reset_acknowledgements": reset_acknowledgements,
                    "active_engine_fingerprints": expected_bindings,
                    "measurement": measurement.to_dict(),
                }
            )
    finally:
        for role, fingerprint in sorted(active, reverse=True):
            driver.stop_engine(role, fingerprint)

    receipt = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "execution_mode": "validation_only",
        "status": "diagnostic_complete",
        "lifecycle_plan_sha256": payload["engine_lifecycle_plan_sha256"],
        "selected_strategy": payload["selected_strategy"],
        "epochs": epochs,
        "summary": _receipt_summary(epochs),
    }
    receipt["lifecycle_execution_receipt_sha256"] = canonical_sha256(receipt)
    return LifecycleExecutionReceipt(receipt)


def audit_lifecycle_execution(
    plan: EngineLifecyclePlan, receipt: LifecycleExecutionReceipt
) -> dict[str, Any]:
    plan_payload = plan.to_dict()
    receipt_payload = receipt.to_dict()
    if receipt_payload["lifecycle_plan_sha256"] != plan_payload["engine_lifecycle_plan_sha256"]:
        raise ValueError("lifecycle execution receipt is bound to another plan")
    if receipt_payload["selected_strategy"] != plan_payload["selected_strategy"]:
        raise ValueError("lifecycle execution strategy differs from the plan")
    if len(receipt_payload["epochs"]) != len(plan_payload["actions"]):
        raise ValueError("lifecycle execution does not cover every planned epoch")
    for schedule, epoch in zip(plan_payload["actions"], receipt_payload["epochs"], strict=True):
        if (
            epoch["run_id"] != schedule["run_id"]
            or epoch["workload_seed"] != schedule["workload_seed"]
            or epoch["planned_actions_sha256"] != canonical_sha256(schedule)
        ):
            raise ValueError("lifecycle epoch receipt differs from the frozen schedule")
        expected_action_receipts = [
            {**action, "status": "acknowledged"}
            for action in (
                *schedule["actions_before_measurement"],
                *schedule["actions_after_measurement"],
            )
        ]
        if epoch["action_receipts"] != expected_action_receipts:
            raise ValueError("lifecycle action receipts differ from the frozen schedule")
        reset_actions = [
            action
            for action in schedule["actions_before_measurement"]
            if action["action"] == "reset_epoch_state"
        ]
        if len(reset_actions) != 1 or set(epoch["reset_acknowledgements"]) != set(
            reset_actions[0]["requirements"]
        ):
            raise ValueError("lifecycle reset receipts differ from the frozen contract")
        if epoch["active_engine_fingerprints"] != _expected_bindings(
            plan_payload, int(schedule["sequence_index"])
        ):
            raise ValueError("lifecycle engine bindings differ from the frozen plan")
    return {
        **receipt_payload["summary"],
        "lifecycle_plan_sha256": receipt_payload["lifecycle_plan_sha256"],
        "receipt_sha256": receipt_payload["lifecycle_execution_receipt_sha256"],
        "validation_only": True,
    }
