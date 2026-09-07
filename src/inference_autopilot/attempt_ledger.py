"""Tamper-evident ledger for logical calibration runs and physical attempts.

This module intentionally has no package-local imports so the campaign launcher can
run it with the Python 3.9 interpreter available on the NPU host.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_ID = re.compile(r"^attempt-(?P<index>\d{3,})$")
_OUTCOMES = {
    "accepted",
    "environment_contaminated",
    "missing_observation",
    "run_failed",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, context: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object: {path}")
    return value


def _relative_path(path: Path, campaign_dir: Path) -> str:
    resolved = path.resolve()
    root = campaign_dir.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError as error:
        raise ValueError(f"attempt artifact must be under campaign directory: {path}") from error


def _validate_embedded_digest(
    raw: Mapping[str, Any], digest_field: str, context: str
) -> str:
    payload = dict(raw)
    digest = payload.pop(digest_field, None)
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise ValueError(f"{context} has no valid {digest_field}")
    if digest != _canonical_sha256(payload):
        raise ValueError(f"{context} {digest_field} does not match its content")
    return digest


def _read_execution_meta(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in values:
            raise ValueError(f"duplicate execution metadata key: {key}")
        values[key] = value
    return values


def _load_records_text(raw: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous_digest: str | None = None
    last_attempt: dict[str, int] = {}
    accepted: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"attempt ledger contains a blank line at {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"attempt ledger line {line_number} must be an object")
        record = dict(value)
        digest = record.pop("record_sha256", None)
        if not isinstance(digest, str) or digest != _canonical_sha256(record):
            raise ValueError(f"attempt ledger hash mismatch at line {line_number}")
        if record.get("sequence_index") != len(records):
            raise ValueError(f"attempt ledger sequence mismatch at line {line_number}")
        if record.get("previous_record_sha256") != previous_digest:
            raise ValueError(f"attempt ledger chain mismatch at line {line_number}")
        run_id = record.get("logical_run_id")
        attempt_index = record.get("attempt_index")
        outcome = record.get("outcome")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"attempt ledger has invalid logical run id at line {line_number}")
        if isinstance(attempt_index, bool) or not isinstance(attempt_index, int):
            raise ValueError(f"attempt ledger has invalid attempt index at line {line_number}")
        expected_attempt = last_attempt.get(run_id, -1) + 1
        if attempt_index != expected_attempt:
            raise ValueError(
                f"attempt index for {run_id} must be {expected_attempt}, got {attempt_index}"
            )
        if outcome not in _OUTCOMES:
            raise ValueError(f"attempt ledger has invalid outcome at line {line_number}")
        if run_id in accepted:
            raise ValueError(f"attempt recorded after accepted outcome for {run_id}")
        if outcome == "accepted":
            accepted.add(run_id)
        last_attempt[run_id] = attempt_index
        previous_digest = digest
        records.append({**record, "record_sha256": digest})
    return records


def load_attempt_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _load_records_text(path.read_text(encoding="utf-8"))


def _artifact_binding(path: Path | None, campaign_dir: Path) -> dict[str, str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise ValueError(f"missing attempt artifact: {path}")
    return {
        "path": _relative_path(path, campaign_dir),
        "sha256": _file_sha256(path),
    }


def _build_attempt_record(
    *,
    existing: Sequence[Mapping[str, Any]],
    campaign_dir: Path,
    bundle_dir: Path,
    logical_run_id: str,
    attempt_index: int,
    outcome: str,
    admission_report: Path,
) -> dict[str, Any]:
    if outcome not in _OUTCOMES:
        raise ValueError(f"unsupported attempt outcome: {outcome}")
    expected_attempt = sum(
        1 for record in existing if record.get("logical_run_id") == logical_run_id
    )
    if attempt_index != expected_attempt:
        raise ValueError(
            f"attempt index for {logical_run_id} must be {expected_attempt}, got {attempt_index}"
        )
    if any(
        record.get("logical_run_id") == logical_run_id
        and record.get("outcome") == "accepted"
        for record in existing
    ):
        raise ValueError(f"logical run already has an accepted attempt: {logical_run_id}")

    manifest_path = bundle_dir / "run-manifest.json"
    manifest = _read_object(manifest_path, "run manifest")
    manifest_digest = _validate_embedded_digest(
        manifest, "run_manifest_sha256", "run manifest"
    )
    manifest_run = manifest.get("run")
    if not isinstance(manifest_run, Mapping) or manifest_run.get("run_id") != logical_run_id:
        raise ValueError("run manifest does not bind the requested logical run id")

    attempt_id = f"attempt-{attempt_index:03d}"
    execution_path = bundle_dir / "execution.meta"
    execution_binding = _artifact_binding(
        execution_path if execution_path.is_file() else None, campaign_dir
    )
    execution = _read_execution_meta(execution_path) if execution_binding else {}
    if execution:
        if execution.get("logical_run_id", execution.get("run_id")) != logical_run_id:
            raise ValueError("execution metadata logical run id mismatch")
        if execution.get("attempt_id") != attempt_id:
            raise ValueError("execution metadata attempt id mismatch")
        actual_outcome = execution.get("run_outcome")
        if outcome == "accepted" and actual_outcome != "success":
            raise ValueError("accepted attempt must have a successful execution outcome")
        if outcome == "environment_contaminated" and actual_outcome != outcome:
            raise ValueError("contaminated attempt must have a matching execution outcome")
    elif outcome in {"accepted", "environment_contaminated"}:
        raise ValueError(f"{outcome} attempt requires execution metadata")

    host_report_path = bundle_dir / "host-interference-report.json"
    host_report_binding = _artifact_binding(
        host_report_path if host_report_path.is_file() else None, campaign_dir
    )
    if host_report_binding:
        host_report = _read_object(host_report_path, "host interference report")
        _validate_embedded_digest(host_report, "report_sha256", "host interference report")
        expected_host_status = "clean" if outcome == "accepted" else None
        if expected_host_status and host_report.get("status") != expected_host_status:
            raise ValueError("accepted attempt requires a clean host interference report")
    elif outcome in {"accepted", "environment_contaminated"}:
        raise ValueError(f"{outcome} attempt requires a host interference report")

    observation_path = bundle_dir / "observation.json"
    observation_binding = _artifact_binding(
        observation_path if observation_path.is_file() else None, campaign_dir
    )
    if observation_binding:
        observation = _read_object(observation_path, "observation")
        if observation.get("run_id") != logical_run_id:
            raise ValueError("observation logical run id mismatch")
        if observation.get("run_manifest_sha256") != manifest_digest:
            raise ValueError("observation run manifest digest mismatch")
        if outcome == "accepted" and observation.get("status") != "success":
            raise ValueError("accepted attempt requires a successful observation")
    elif outcome == "accepted":
        raise ValueError("accepted attempt requires an observation")

    admission_binding = _artifact_binding(admission_report, campaign_dir)
    admission = _read_object(admission_report, "admission report")
    _validate_embedded_digest(admission, "report_sha256", "admission report")
    if admission.get("status") != "clean":
        raise ValueError("a launched attempt must be bound to a clean admission report")

    payload = {
        "schema_version": "1.0",
        "sequence_index": len(existing),
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "logical_run_id": logical_run_id,
        "attempt_id": attempt_id,
        "attempt_index": attempt_index,
        "outcome": outcome,
        "bundle_path": _relative_path(bundle_dir, campaign_dir),
        "run_manifest_sha256": manifest_digest,
        "run_manifest_file_sha256": _file_sha256(manifest_path),
        "execution_meta": execution_binding,
        "host_interference_report": host_report_binding,
        "observation": observation_binding,
        "admission_report": admission_binding,
        "previous_record_sha256": (
            existing[-1]["record_sha256"] if existing else None
        ),
    }
    return {**payload, "record_sha256": _canonical_sha256(payload)}


def append_attempt(
    ledger_path: Path,
    *,
    campaign_dir: Path,
    bundle_dir: Path,
    logical_run_id: str,
    attempt_index: int,
    outcome: str,
    admission_report: Path,
) -> dict[str, Any]:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        existing = _load_records_text(stream.read())
        record = _build_attempt_record(
            existing=existing,
            campaign_dir=campaign_dir,
            bundle_dir=bundle_dir,
            logical_run_id=logical_run_id,
            attempt_index=attempt_index,
            outcome=outcome,
            admission_report=admission_report,
        )
        stream.seek(0, 2)
        stream.write(_canonical_json(record) + "\n")
        stream.flush()
        import os

        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return record


def _verify_binding(
    binding: Mapping[str, Any] | None,
    *,
    campaign_dir: Path,
    context: str,
    required: bool,
) -> Path | None:
    if binding is None:
        if required:
            raise ValueError(f"missing {context} binding")
        return None
    path_value = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise ValueError(f"invalid {context} binding")
    path = campaign_dir / path_value
    if not path.is_file() or _file_sha256(path) != digest:
        raise ValueError(f"{context} artifact hash mismatch: {path_value}")
    return path


def audit_attempt_ledger(
    ledger_path: Path,
    *,
    campaign_dir: Path,
    plan_path: Path,
    require_complete: bool = True,
) -> dict[str, Any]:
    records = load_attempt_ledger(ledger_path)
    plan = _read_object(plan_path, "calibration plan")
    raw_runs = plan.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("calibration plan has no runs")
    expected_runs: list[dict[str, Any]] = []
    for raw in raw_runs:
        if not isinstance(raw, dict) or not isinstance(raw.get("run_id"), str):
            raise ValueError("calibration plan contains an invalid run")
        expected_runs.append(raw)
    expected_ids = [run["run_id"] for run in expected_runs]
    expected_by_id = {run["run_id"]: run for run in expected_runs}
    plan_digest = _canonical_sha256(plan)

    observed_ids = {str(record["logical_run_id"]) for record in records}
    unexpected = sorted(observed_ids - set(expected_ids))
    if unexpected:
        raise ValueError(f"attempt ledger contains unexpected logical runs: {unexpected}")
    plan_indexes = [expected_ids.index(str(record["logical_run_id"])) for record in records]
    if plan_indexes != sorted(plan_indexes):
        raise ValueError("attempt records do not preserve logical plan order")

    accepted_ids: list[str] = []
    for record in records:
        run_id = str(record["logical_run_id"])
        bundle_dir = campaign_dir / str(record["bundle_path"])
        manifest_path = bundle_dir / "run-manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing recorded run manifest: {manifest_path}")
        if _file_sha256(manifest_path) != record.get("run_manifest_file_sha256"):
            raise ValueError(f"run manifest file hash mismatch for {run_id}")
        manifest = _read_object(manifest_path, "run manifest")
        manifest_digest = _validate_embedded_digest(
            manifest, "run_manifest_sha256", "run manifest"
        )
        if manifest_digest != record.get("run_manifest_sha256"):
            raise ValueError(f"run manifest digest mismatch for {run_id}")
        if manifest.get("run") != expected_by_id[run_id]:
            raise ValueError(f"run manifest does not match planned logical run: {run_id}")
        if manifest.get("plan_sha256") != plan_digest:
            raise ValueError(f"run manifest plan binding mismatch for {run_id}")

        required = record["outcome"] in {"accepted", "environment_contaminated"}
        execution_path = _verify_binding(
            record.get("execution_meta"),
            campaign_dir=campaign_dir,
            context="execution metadata",
            required=required,
        )
        host_path = _verify_binding(
            record.get("host_interference_report"),
            campaign_dir=campaign_dir,
            context="host interference report",
            required=required,
        )
        observation_path = _verify_binding(
            record.get("observation"),
            campaign_dir=campaign_dir,
            context="observation",
            required=record["outcome"] == "accepted",
        )
        admission_path = _verify_binding(
            record.get("admission_report"),
            campaign_dir=campaign_dir,
            context="admission report",
            required=True,
        )
        if admission_path is not None:
            admission = _read_object(admission_path, "admission report")
            _validate_embedded_digest(admission, "report_sha256", "admission report")
            if admission.get("status") != "clean":
                raise ValueError(f"attempt has non-clean admission report: {run_id}")
        if execution_path is not None:
            execution = _read_execution_meta(execution_path)
            if execution.get("logical_run_id", execution.get("run_id")) != run_id:
                raise ValueError(f"execution metadata run mismatch for {run_id}")
            if execution.get("attempt_id") != record.get("attempt_id"):
                raise ValueError(f"execution metadata attempt mismatch for {run_id}")
        if host_path is not None:
            host = _read_object(host_path, "host interference report")
            _validate_embedded_digest(host, "report_sha256", "host interference report")
        if observation_path is not None:
            observation = _read_object(observation_path, "observation")
            if observation.get("run_id") != run_id:
                raise ValueError(f"observation run mismatch for {run_id}")
        if record["outcome"] == "accepted":
            if host_path is None or _read_object(host_path, "host report").get("status") != "clean":
                raise ValueError(f"accepted attempt has non-clean host report: {run_id}")
            if (
                observation_path is None
                or _read_object(observation_path, "observation").get("status") != "success"
            ):
                raise ValueError(f"accepted attempt has non-success observation: {run_id}")
            accepted_ids.append(run_id)

    missing = [run_id for run_id in expected_ids if run_id not in accepted_ids]
    if accepted_ids != expected_ids[: len(accepted_ids)]:
        raise ValueError("accepted logical runs do not form a plan prefix")
    active_ids = sorted(observed_ids - set(accepted_ids), key=expected_ids.index)
    if len(active_ids) > 1 or (
        active_ids and active_ids[0] != expected_ids[len(accepted_ids)]
    ):
        raise ValueError("partial ledger contains attempts beyond the next logical run")
    if require_complete and missing:
        raise ValueError(f"logical runs have no accepted attempt: {missing}")
    if require_complete and accepted_ids != expected_ids:
        raise ValueError("accepted logical runs do not preserve plan order")

    payload = {
        "schema_version": "1.0",
        "status": "complete" if not missing else "partial",
        "plan_file_sha256": _file_sha256(plan_path),
        "ledger_file_sha256": (
            _file_sha256(ledger_path)
            if ledger_path.is_file()
            else hashlib.sha256(b"").hexdigest()
        ),
        "expected_run_ids": expected_ids,
        "accepted_run_ids": accepted_ids,
        "missing_run_ids": missing,
        "active_run_id": active_ids[0] if active_ids else None,
        "attempt_count": len(records),
        "contaminated_attempt_count": sum(
            record["outcome"] == "environment_contaminated" for record in records
        ),
        "final_record_sha256": records[-1]["record_sha256"] if records else None,
    }
    return {**payload, "report_sha256": _canonical_sha256(payload)}


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="attempt_ledger.py")
    subparsers = parser.add_subparsers(dest="command", required=True)
    append_parser = subparsers.add_parser("append")
    append_parser.add_argument("ledger", type=Path)
    append_parser.add_argument("--campaign-dir", type=Path, required=True)
    append_parser.add_argument("--bundle-dir", type=Path, required=True)
    append_parser.add_argument("--logical-run-id", required=True)
    append_parser.add_argument("--attempt-index", type=int, required=True)
    append_parser.add_argument("--outcome", choices=sorted(_OUTCOMES), required=True)
    append_parser.add_argument("--admission-report", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("ledger", type=Path)
    audit_parser.add_argument("--campaign-dir", type=Path, required=True)
    audit_parser.add_argument("--plan", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--allow-incomplete", action="store_true")
    state_parser = subparsers.add_parser("state")
    state_parser.add_argument("ledger", type=Path)
    state_parser.add_argument("--logical-run-id", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "append":
            record = append_attempt(
                args.ledger,
                campaign_dir=args.campaign_dir,
                bundle_dir=args.bundle_dir,
                logical_run_id=args.logical_run_id,
                attempt_index=args.attempt_index,
                outcome=args.outcome,
                admission_report=args.admission_report,
            )
            print(record["record_sha256"])
            return 0
        if args.command == "state":
            records = load_attempt_ledger(args.ledger)
            matching = [
                record
                for record in records
                if record["logical_run_id"] == args.logical_run_id
            ]
            print(
                _canonical_json(
                    {
                        "logical_run_id": args.logical_run_id,
                        "attempt_count": len(matching),
                        "accepted": any(
                            record["outcome"] == "accepted" for record in matching
                        ),
                    }
                )
            )
            return 0
        report = audit_attempt_ledger(
            args.ledger,
            campaign_dir=args.campaign_dir,
            plan_path=args.plan,
            require_complete=not args.allow_incomplete,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if args.command != "audit":
            print(f"attempt ledger error: {error}", file=sys.stderr)
            return 2
        payload = {
            "schema_version": "1.0",
            "status": "invalid",
            "error": str(error),
        }
        report = {**payload, "report_sha256": _canonical_sha256(payload)}
        _write_report(args.output, report)
        print(report["status"])
        return 3
    _write_report(args.output, report)
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
