"""Low-overhead host telemetry and fail-closed interference assessment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


_PROCESS_ROW = re.compile(
    r"^\|\s*(?P<device>\d+)\s+\d+\s*\|\s*(?P<pid>\d+)\s*\|"
    r"\s*(?P<name>[^|]+?)\s*\|\s*(?P<memory>\d+)\s*\|$"
)
_DEVICE_HEADER = re.compile(r"^(?P<device>\d+)\s+(?P<name>\S+)$")
_HBM_USAGE = re.compile(r"(?P<used>\d+)\s*/\s*(?P<total>\d+)\s*$")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = quantile * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_npu_smi_info(output: str) -> list[dict[str, Any]]:
    """Parse the stable device and process fields from ``npu-smi info``."""

    devices: dict[int, dict[str, Any]] = {}
    pending_device: int | None = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        process_match = _PROCESS_ROW.match(line)
        if process_match:
            device_id = int(process_match.group("device"))
            device = devices.setdefault(device_id, {"physical_device_id": device_id})
            device.setdefault("processes", []).append(
                {
                    "pid": int(process_match.group("pid")),
                    "name": process_match.group("name").strip(),
                    "memory_mb": int(process_match.group("memory")),
                }
            )
            continue
        if not line.startswith("|") or line.startswith("| NPU") or line.startswith("| Chip"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) != 3:
            continue
        header_match = _DEVICE_HEADER.match(columns[0])
        if header_match and columns[1] in {"OK", "Warning", "Alarm", "Critical"}:
            device_id = int(header_match.group("device"))
            numeric = re.findall(r"-?\d+(?:\.\d+)?", columns[2])
            if len(numeric) >= 2:
                device = devices.setdefault(device_id, {"physical_device_id": device_id})
                device.update(
                    {
                        "name": header_match.group("name"),
                        "health": columns[1],
                        "power_watts": float(numeric[0]),
                        "temperature_c": float(numeric[1]),
                    }
                )
                pending_device = device_id
            continue
        if pending_device is None:
            continue
        hbm_match = _HBM_USAGE.search(columns[2])
        numeric = re.findall(r"-?\d+(?:\.\d+)?", columns[2])
        if hbm_match and numeric:
            devices[pending_device].update(
                {
                    "aicore_percent": float(numeric[0]),
                    "hbm_used_mb": int(hbm_match.group("used")),
                    "hbm_total_mb": int(hbm_match.group("total")),
                }
            )
            pending_device = None

    parsed: list[dict[str, Any]] = []
    for device_id in sorted(devices):
        device = devices[device_id]
        processes = sorted(device.get("processes", []), key=lambda item: item["pid"])
        device["processes"] = processes
        device["process_memory_mb"] = sum(item["memory_mb"] for item in processes)
        parsed.append(device)
    return parsed


def _read_cpu() -> dict[str, int]:
    first_line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    fields = [int(value) for value in first_line.split()[1:]]
    if len(fields) < 5:
        raise ValueError("/proc/stat does not contain the expected aggregate CPU fields")
    idle = fields[3]
    iowait = fields[4]
    return {
        "total_jiffies": sum(fields),
        "idle_jiffies": idle,
        "iowait_jiffies": iowait,
        "logical_cpu_count": sum(
            1
            for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines()
            if re.match(r"^cpu\d+\s", line)
        ),
    }


def _read_load() -> dict[str, float | int]:
    fields = Path("/proc/loadavg").read_text(encoding="utf-8").split()
    runnable, processes = fields[3].split("/", maxsplit=1)
    return {
        "load1": float(fields[0]),
        "load5": float(fields[1]),
        "load15": float(fields[2]),
        "runnable_processes": int(runnable),
        "total_processes": int(processes),
    }


def _read_memory() -> dict[str, int]:
    wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, _, remainder = line.partition(":")
        if name in wanted:
            values[name] = int(remainder.strip().split()[0])
    if {"MemTotal", "MemAvailable"} - set(values):
        raise ValueError("/proc/meminfo is missing total or available memory")
    return {
        "total_kib": values["MemTotal"],
        "available_kib": values["MemAvailable"],
        "swap_total_kib": values.get("SwapTotal", 0),
        "swap_free_kib": values.get("SwapFree", 0),
    }


def capture_host_sample(npu_smi_path: Path) -> dict[str, Any]:
    started = time.time()
    try:
        completed = subprocess.run(
            [str(npu_smi_path), "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        npu_output = completed.stdout + completed.stderr
        devices = parse_npu_smi_info(npu_output)
        npu_status = "success" if completed.returncode == 0 and devices else "failed"
    except (OSError, subprocess.SubprocessError) as error:
        npu_output = f"{type(error).__name__}: {error}"
        devices = []
        npu_status = "failed"
    return {
        "schema_version": "1.0",
        "captured_at_unix": started,
        "capture_duration_seconds": time.time() - started,
        "cpu": _read_cpu(),
        "load": _read_load(),
        "memory": _read_memory(),
        "npu": {
            "status": npu_status,
            "raw_sha256": hashlib.sha256(npu_output.encode("utf-8")).hexdigest(),
            "devices": devices,
        },
    }


@dataclass(frozen=True)
class HostInterferencePolicy:
    minimum_samples: int = 3
    max_sample_gap_seconds: float = 30.0
    max_host_cpu_busy_fraction: float = 0.90
    max_host_iowait_fraction: float = 0.15
    max_run_queue_per_cpu: float = 1.0
    min_memory_available_fraction: float = 0.05
    reject_sibling_process_churn: bool = True
    require_npu_telemetry: bool = True
    require_target_idle: bool = False

    def __post_init__(self) -> None:
        if self.minimum_samples < 2:
            raise ValueError("host telemetry requires at least two samples")
        if self.max_sample_gap_seconds <= 0:
            raise ValueError("maximum sample gap must be positive")
        for name in (
            "max_host_cpu_busy_fraction",
            "max_host_iowait_fraction",
            "min_memory_available_fraction",
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.max_run_queue_per_cpu <= 0:
            raise ValueError("maximum run queue per CPU must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignInterferencePolicy:
    max_host_cpu_busy_p95_range: float = 0.20
    max_host_iowait_p95_range: float = 0.10
    max_run_queue_per_cpu_p95_range: float = 0.25
    max_memory_available_fraction_range: float = 0.10
    max_sibling_aicore_mean_range_percent: float = 20.0
    max_target_initial_temperature_range_c: float = 10.0
    max_target_initial_power_range_watts: float = 30.0
    reject_sibling_process_changes_between_runs: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_host_cpu_busy_p95_range",
            "max_host_iowait_p95_range",
            "max_run_queue_per_cpu_p95_range",
            "max_memory_available_fraction_range",
            "max_sibling_aicore_mean_range_percent",
            "max_target_initial_temperature_range_c",
            "max_target_initial_power_range_watts",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _device_map(sample: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    npu = sample.get("npu", {})
    devices = npu.get("devices", []) if isinstance(npu, Mapping) else []
    return {
        int(device["physical_device_id"]): device
        for device in devices
        if isinstance(device, Mapping) and "physical_device_id" in device
    }


def _process_fingerprint(device: Mapping[str, Any]) -> tuple[tuple[int, str], ...]:
    processes = device.get("processes", [])
    return tuple(
        sorted(
            (int(process["pid"]), str(process.get("name", "")))
            for process in processes
            if isinstance(process, Mapping) and "pid" in process
        )
    )


def _load_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"host telemetry line {line_number} must be an object")
        samples.append(value)
    return sorted(samples, key=lambda item: float(item["captured_at_unix"]))


def assess_host_telemetry(
    samples: Sequence[Mapping[str, Any]],
    *,
    target_npu_id: int,
    policy: HostInterferencePolicy,
    telemetry_sha256: str,
) -> dict[str, Any]:
    """Assess whether a benchmark ran under stable, observable host conditions."""

    hard_findings: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    timestamps = [float(sample["captured_at_unix"]) for sample in samples]
    gaps = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    successful_npu_samples = [
        sample
        for sample in samples
        if isinstance(sample.get("npu"), Mapping)
        and sample["npu"].get("status") == "success"
    ]
    if len(samples) < policy.minimum_samples:
        hard_findings.append(
            {
                "code": "insufficient_samples",
                "observed": len(samples),
                "threshold": policy.minimum_samples,
            }
        )
    if gaps and max(gaps) > policy.max_sample_gap_seconds:
        hard_findings.append(
            {
                "code": "telemetry_gap",
                "observed": max(gaps),
                "threshold": policy.max_sample_gap_seconds,
            }
        )
    if policy.require_npu_telemetry and len(successful_npu_samples) != len(samples):
        hard_findings.append(
            {
                "code": "missing_npu_telemetry",
                "observed": len(successful_npu_samples),
                "threshold": len(samples),
            }
        )

    cpu_busy: list[float] = []
    cpu_iowait: list[float] = []
    for previous, current in zip(samples, samples[1:]):
        previous_cpu = previous["cpu"]
        current_cpu = current["cpu"]
        total_delta = int(current_cpu["total_jiffies"]) - int(previous_cpu["total_jiffies"])
        if total_delta <= 0:
            continue
        idle_delta = int(current_cpu["idle_jiffies"]) - int(previous_cpu["idle_jiffies"])
        iowait_delta = int(current_cpu["iowait_jiffies"]) - int(previous_cpu["iowait_jiffies"])
        cpu_busy.append(max(0.0, min(1.0, 1.0 - idle_delta / total_delta)))
        cpu_iowait.append(max(0.0, min(1.0, iowait_delta / total_delta)))
    p95_cpu = _percentile(cpu_busy, 0.95)
    p95_iowait = _percentile(cpu_iowait, 0.95)
    if p95_cpu is not None and p95_cpu > policy.max_host_cpu_busy_fraction:
        hard_findings.append(
            {
                "code": "host_cpu_pressure",
                "observed": p95_cpu,
                "threshold": policy.max_host_cpu_busy_fraction,
            }
        )
    if p95_iowait is not None and p95_iowait > policy.max_host_iowait_fraction:
        hard_findings.append(
            {
                "code": "host_iowait_pressure",
                "observed": p95_iowait,
                "threshold": policy.max_host_iowait_fraction,
            }
        )

    queue_ratios: list[float] = []
    memory_fractions: list[float] = []
    for sample in samples:
        cpu_count = max(1, int(sample["cpu"]["logical_cpu_count"]))
        queue_ratios.append(float(sample["load"]["runnable_processes"]) / cpu_count)
        memory = sample["memory"]
        memory_fractions.append(float(memory["available_kib"]) / float(memory["total_kib"]))
    p95_queue = _percentile(queue_ratios, 0.95)
    minimum_memory = min(memory_fractions) if memory_fractions else None
    if p95_queue is not None and p95_queue > policy.max_run_queue_per_cpu:
        hard_findings.append(
            {
                "code": "host_run_queue_pressure",
                "observed": p95_queue,
                "threshold": policy.max_run_queue_per_cpu,
            }
        )
    if minimum_memory is not None and minimum_memory < policy.min_memory_available_fraction:
        hard_findings.append(
            {
                "code": "host_memory_pressure",
                "observed": minimum_memory,
                "threshold": policy.min_memory_available_fraction,
            }
        )

    sibling_fingerprints: dict[int, set[tuple[tuple[int, str], ...]]] = {}
    sibling_aicore: dict[int, list[float]] = {}
    observed_device_ids: set[int] = set()
    target_busy_sample_count = 0
    for sample in successful_npu_samples:
        for device_id, device in _device_map(sample).items():
            observed_device_ids.add(device_id)
            if device_id == target_npu_id:
                if _process_fingerprint(device):
                    target_busy_sample_count += 1
                continue
            sibling_fingerprints.setdefault(device_id, set()).add(_process_fingerprint(device))
            if "aicore_percent" in device:
                sibling_aicore.setdefault(device_id, []).append(float(device["aicore_percent"]))
    churned_devices = sorted(
        device_id
        for device_id, fingerprints in sibling_fingerprints.items()
        if len(fingerprints) > 1
    )
    if policy.reject_sibling_process_churn and churned_devices:
        hard_findings.append(
            {
                "code": "sibling_npu_process_churn",
                "observed": churned_devices,
                "threshold": "no process identity changes",
            }
        )
    if successful_npu_samples and target_npu_id not in observed_device_ids:
        hard_findings.append(
            {
                "code": "target_npu_missing",
                "observed": sorted(observed_device_ids),
                "threshold": target_npu_id,
            }
        )
    if policy.require_target_idle and target_busy_sample_count:
        hard_findings.append(
            {
                "code": "target_npu_busy",
                "observed": target_busy_sample_count,
                "threshold": 0,
            }
        )
    for device_id, values in sorted(sibling_aicore.items()):
        if values and max(values) > 0:
            warnings.append(
                {
                    "code": "sibling_npu_active",
                    "device_id": device_id,
                    "maximum_aicore_percent": max(values),
                    "aicore_range_percent": max(values) - min(values),
                }
            )

    sibling_activity = {
        str(device_id): {
            "aicore_percent_minimum": min(values) if values else None,
            "aicore_percent_maximum": max(values) if values else None,
            "aicore_percent_mean": sum(values) / len(values) if values else None,
            "process_fingerprint_sha256": (
                _canonical_sha256(
                    {
                        "processes": [list(process) for process in next(iter(fingerprints))]
                    }
                )
                if len(fingerprints) == 1
                else None
            ),
        }
        for device_id, values in sorted(sibling_aicore.items())
        for fingerprints in (sibling_fingerprints.get(device_id, set()),)
    }
    target_initial: dict[str, float] = {}
    if successful_npu_samples:
        target = _device_map(successful_npu_samples[0]).get(target_npu_id, {})
        for source_name, result_name in (
            ("temperature_c", "temperature_c"),
            ("power_watts", "power_watts"),
            ("hbm_used_mb", "hbm_used_mb"),
        ):
            value = target.get(source_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                target_initial[result_name] = float(value)

    insufficient_codes = {
        "insufficient_samples",
        "telemetry_gap",
        "missing_npu_telemetry",
        "target_npu_missing",
    }
    status = (
        "insufficient_telemetry"
        if any(finding["code"] in insufficient_codes for finding in hard_findings)
        else "contaminated"
        if hard_findings
        else "clean"
    )
    payload = {
        "schema_version": "1.0",
        "status": status,
        "target_npu_id": target_npu_id,
        "telemetry_sha256": telemetry_sha256,
        "policy": policy.to_dict(),
        "summary": {
            "sample_count": len(samples),
            "duration_seconds": timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0.0,
            "maximum_sample_gap_seconds": max(gaps) if gaps else None,
            "successful_npu_sample_count": len(successful_npu_samples),
            "host_cpu_busy_p95": p95_cpu,
            "host_iowait_p95": p95_iowait,
            "run_queue_per_cpu_p95": p95_queue,
            "minimum_memory_available_fraction": minimum_memory,
            "observed_npu_ids": sorted(observed_device_ids),
            "target_busy_sample_count": target_busy_sample_count,
            "sibling_process_churn_device_ids": churned_devices,
            "sibling_npu_activity": sibling_activity,
            "target_npu_initial": target_initial,
        },
        "hard_findings": hard_findings,
        "warnings": warnings,
    }
    return {**payload, "report_sha256": _canonical_sha256(payload)}


def _validated_report(raw: Mapping[str, Any]) -> dict[str, Any]:
    report = dict(raw)
    digest = report.pop("report_sha256", None)
    if not isinstance(digest, str) or digest != _canonical_sha256(report):
        raise ValueError("host interference report SHA256 does not match its content")
    return dict(raw)


def assess_campaign_host_reports(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    expected_run_ids: Sequence[str],
    plan_sha256: str,
    policy: CampaignInterferencePolicy,
) -> dict[str, Any]:
    """Reject host-state changes across otherwise clean ordered runs."""

    hard_findings: list[dict[str, Any]] = []
    expected = tuple(expected_run_ids)
    missing = sorted(set(expected) - set(reports))
    unexpected = sorted(set(reports) - set(expected))
    if missing:
        hard_findings.append(
            {"code": "missing_run_host_report", "observed": missing, "threshold": "none"}
        )
    if unexpected:
        hard_findings.append(
            {"code": "unexpected_run_host_report", "observed": unexpected, "threshold": "none"}
        )

    valid_reports: dict[str, Mapping[str, Any]] = {}
    invalid_reports: list[str] = []
    for run_id, report in reports.items():
        try:
            valid_reports[run_id] = _validated_report(report)
        except (TypeError, ValueError):
            invalid_reports.append(run_id)
    if invalid_reports:
        hard_findings.append(
            {
                "code": "invalid_run_host_report",
                "observed": sorted(invalid_reports),
                "threshold": "valid content hash",
            }
        )

    nonclean = {
        run_id: report.get("status")
        for run_id, report in valid_reports.items()
        if report.get("status") != "clean"
    }
    if nonclean:
        hard_findings.append(
            {"code": "nonclean_run_host_report", "observed": nonclean, "threshold": "clean"}
        )

    def summary_values(name: str) -> list[float]:
        values: list[float] = []
        for run_id in expected:
            report = valid_reports.get(run_id)
            if report is None or not isinstance(report.get("summary"), Mapping):
                continue
            value = report["summary"].get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        return values

    range_specs = (
        ("host_cpu_busy_p95", policy.max_host_cpu_busy_p95_range, "host_cpu_shift"),
        ("host_iowait_p95", policy.max_host_iowait_p95_range, "host_iowait_shift"),
        (
            "run_queue_per_cpu_p95",
            policy.max_run_queue_per_cpu_p95_range,
            "host_run_queue_shift",
        ),
        (
            "minimum_memory_available_fraction",
            policy.max_memory_available_fraction_range,
            "host_memory_shift",
        ),
    )
    observed_ranges: dict[str, float | None] = {}
    for metric, threshold, code in range_specs:
        values = summary_values(metric)
        observed_range = max(values) - min(values) if values else None
        observed_ranges[metric] = observed_range
        if observed_range is not None and observed_range > threshold:
            hard_findings.append(
                {"code": code, "observed": observed_range, "threshold": threshold}
            )

    target_initial_ranges: dict[str, float | None] = {}
    for metric, threshold, code in (
        (
            "temperature_c",
            policy.max_target_initial_temperature_range_c,
            "target_npu_initial_temperature_shift",
        ),
        (
            "power_watts",
            policy.max_target_initial_power_range_watts,
            "target_npu_initial_power_shift",
        ),
    ):
        values: list[float] = []
        for run_id in expected:
            report = valid_reports.get(run_id)
            summary = report.get("summary", {}) if report is not None else {}
            target = summary.get("target_npu_initial", {}) if isinstance(summary, Mapping) else {}
            value = target.get(metric) if isinstance(target, Mapping) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        observed_range = max(values) - min(values) if values else None
        target_initial_ranges[metric] = observed_range
        if observed_range is not None and observed_range > threshold:
            hard_findings.append(
                {"code": code, "observed": observed_range, "threshold": threshold}
            )

    sibling_processes: dict[str, set[str]] = {}
    sibling_aicore_means: dict[str, list[float]] = {}
    for report in valid_reports.values():
        summary = report.get("summary", {})
        activity = summary.get("sibling_npu_activity", {}) if isinstance(summary, Mapping) else {}
        if not isinstance(activity, Mapping):
            continue
        for device_id, device in activity.items():
            if not isinstance(device, Mapping):
                continue
            fingerprint = device.get("process_fingerprint_sha256")
            if isinstance(fingerprint, str):
                sibling_processes.setdefault(str(device_id), set()).add(fingerprint)
            mean = device.get("aicore_percent_mean")
            if isinstance(mean, (int, float)) and not isinstance(mean, bool):
                sibling_aicore_means.setdefault(str(device_id), []).append(float(mean))

    changed_sibling_processes = sorted(
        device_id for device_id, fingerprints in sibling_processes.items() if len(fingerprints) > 1
    )
    if policy.reject_sibling_process_changes_between_runs and changed_sibling_processes:
        hard_findings.append(
            {
                "code": "sibling_npu_process_change_between_runs",
                "observed": changed_sibling_processes,
                "threshold": "stable process identity",
            }
        )
    sibling_aicore_ranges = {
        device_id: max(values) - min(values)
        for device_id, values in sorted(sibling_aicore_means.items())
        if values
    }
    shifted_sibling_activity = {
        device_id: value
        for device_id, value in sibling_aicore_ranges.items()
        if value > policy.max_sibling_aicore_mean_range_percent
    }
    if shifted_sibling_activity:
        hard_findings.append(
            {
                "code": "sibling_npu_activity_shift_between_runs",
                "observed": shifted_sibling_activity,
                "threshold": policy.max_sibling_aicore_mean_range_percent,
            }
        )

    insufficient_codes = {
        "missing_run_host_report",
        "unexpected_run_host_report",
        "invalid_run_host_report",
        "nonclean_run_host_report",
    }
    status = (
        "insufficient_telemetry"
        if any(finding["code"] in insufficient_codes for finding in hard_findings)
        else "contaminated"
        if hard_findings
        else "clean"
    )
    payload = {
        "schema_version": "1.0",
        "status": status,
        "plan_sha256": plan_sha256,
        "expected_run_ids": list(expected),
        "run_report_sha256": {
            run_id: str(report.get("report_sha256", ""))
            for run_id, report in sorted(valid_reports.items())
        },
        "policy": policy.to_dict(),
        "summary": {
            "run_count": len(valid_reports),
            "cross_run_ranges": observed_ranges,
            "sibling_process_change_device_ids": changed_sibling_processes,
            "sibling_aicore_mean_ranges_percent": sibling_aicore_ranges,
            "target_npu_initial_ranges": target_initial_ranges,
        },
        "hard_findings": hard_findings,
    }
    return {**payload, "report_sha256": _canonical_sha256(payload)}


def assess_campaign_host_report_files(
    campaign_dir: Path,
    *,
    plan_path: Path,
    policy: CampaignInterferencePolicy,
) -> dict[str, Any]:
    plan_raw = json.loads(plan_path.read_text(encoding="utf-8"))
    runs = plan_raw.get("runs", []) if isinstance(plan_raw, Mapping) else []
    expected_run_ids = tuple(str(run["run_id"]) for run in runs)
    reports: dict[str, Mapping[str, Any]] = {}
    for run_id in expected_run_ids:
        report_path = campaign_dir / run_id / "host-interference-report.json"
        if report_path.is_file():
            value = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                telemetry_path = campaign_dir / run_id / "host-telemetry.jsonl"
                if (
                    not telemetry_path.is_file()
                    or hashlib.sha256(telemetry_path.read_bytes()).hexdigest()
                    != value.get("telemetry_sha256")
                ):
                    value = {**value, "telemetry_sha256": "mismatch"}
                reports[run_id] = value
    return assess_campaign_host_reports(
        reports,
        expected_run_ids=expected_run_ids,
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        policy=policy,
    )


def assess_host_telemetry_file(
    telemetry_path: Path,
    *,
    target_npu_id: int,
    policy: HostInterferencePolicy,
) -> dict[str, Any]:
    raw = telemetry_path.read_bytes()
    return assess_host_telemetry(
        _load_samples(telemetry_path),
        target_npu_id=target_npu_id,
        policy=policy,
        telemetry_sha256=hashlib.sha256(raw).hexdigest(),
    )


def monitor(output: Path, *, interval_seconds: float, npu_smi_path: Path) -> None:
    if interval_seconds <= 0:
        raise ValueError("monitor interval must be positive")
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", buffering=1) as stream:
        while True:
            stream.write(json.dumps(capture_host_sample(npu_smi_path), sort_keys=True) + "\n")
            if stop:
                break
            deadline = time.monotonic() + interval_seconds
            while not stop and time.monotonic() < deadline:
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))


def capture_window(
    output: Path,
    *,
    duration_seconds: float,
    interval_seconds: float,
    npu_smi_path: Path,
    capture: Callable[[Path], Mapping[str, Any]] = capture_host_sample,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Capture an immediate, bounded host-stability window without a daemon."""

    if duration_seconds <= 0:
        raise ValueError("admission window duration must be positive")
    if interval_seconds <= 0:
        raise ValueError("admission window interval must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    with output.open("x", encoding="utf-8", buffering=1) as stream:
        while True:
            stream.write(json.dumps(dict(capture(npu_smi_path)), sort_keys=True) + "\n")
            remaining = duration_seconds - (monotonic() - started)
            if remaining <= 0:
                break
            sleep(min(interval_seconds, remaining))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m inference_autopilot.host_interference")
    subparsers = parser.add_subparsers(dest="command", required=True)
    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--output", type=Path, required=True)
    monitor_parser.add_argument("--interval-seconds", type=float, default=10.0)
    monitor_parser.add_argument("--npu-smi", type=Path, default=Path("/usr/local/bin/npu-smi"))
    window_parser = subparsers.add_parser("check-window")
    window_parser.add_argument("--output-telemetry", type=Path, required=True)
    window_parser.add_argument("--output", type=Path, required=True)
    window_parser.add_argument("--target-npu", type=int, required=True)
    window_parser.add_argument("--duration-seconds", type=float, default=35.0)
    window_parser.add_argument("--interval-seconds", type=float, default=10.0)
    window_parser.add_argument("--npu-smi", type=Path, default=Path("/usr/local/bin/npu-smi"))
    window_parser.add_argument("--minimum-samples", type=int, default=3)
    window_parser.add_argument("--max-sample-gap-seconds", type=float, default=30.0)
    window_parser.add_argument("--max-host-cpu-busy-fraction", type=float, default=0.90)
    window_parser.add_argument("--max-host-iowait-fraction", type=float, default=0.15)
    window_parser.add_argument("--max-run-queue-per-cpu", type=float, default=1.0)
    window_parser.add_argument("--min-memory-available-fraction", type=float, default=0.05)
    assess_parser = subparsers.add_parser("assess")
    assess_parser.add_argument("telemetry", type=Path)
    assess_parser.add_argument("--target-npu", type=int, required=True)
    assess_parser.add_argument("--output", type=Path, required=True)
    assess_parser.add_argument("--minimum-samples", type=int, default=3)
    assess_parser.add_argument("--max-sample-gap-seconds", type=float, default=30.0)
    assess_parser.add_argument("--max-host-cpu-busy-fraction", type=float, default=0.90)
    assess_parser.add_argument("--max-host-iowait-fraction", type=float, default=0.15)
    assess_parser.add_argument("--max-run-queue-per-cpu", type=float, default=1.0)
    assess_parser.add_argument("--min-memory-available-fraction", type=float, default=0.05)
    campaign_parser = subparsers.add_parser("assess-campaign")
    campaign_parser.add_argument("campaign_dir", type=Path)
    campaign_parser.add_argument("--plan", type=Path, required=True)
    campaign_parser.add_argument("--output", type=Path, required=True)
    campaign_parser.add_argument("--max-host-cpu-busy-p95-range", type=float, default=0.20)
    campaign_parser.add_argument("--max-host-iowait-p95-range", type=float, default=0.10)
    campaign_parser.add_argument("--max-run-queue-per-cpu-p95-range", type=float, default=0.25)
    campaign_parser.add_argument(
        "--max-memory-available-fraction-range", type=float, default=0.10
    )
    campaign_parser.add_argument(
        "--max-sibling-aicore-mean-range-percent", type=float, default=20.0
    )
    campaign_parser.add_argument(
        "--max-target-initial-temperature-range-c", type=float, default=10.0
    )
    campaign_parser.add_argument(
        "--max-target-initial-power-range-watts", type=float, default=30.0
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "monitor":
        monitor(args.output, interval_seconds=args.interval_seconds, npu_smi_path=args.npu_smi)
        return 0
    if args.command in {"assess", "check-window"}:
        policy = HostInterferencePolicy(
            minimum_samples=args.minimum_samples,
            max_sample_gap_seconds=args.max_sample_gap_seconds,
            max_host_cpu_busy_fraction=args.max_host_cpu_busy_fraction,
            max_host_iowait_fraction=args.max_host_iowait_fraction,
            max_run_queue_per_cpu=args.max_run_queue_per_cpu,
            min_memory_available_fraction=args.min_memory_available_fraction,
            require_target_idle=args.command == "check-window",
        )
        telemetry = args.telemetry if args.command == "assess" else args.output_telemetry
        if args.command == "check-window":
            capture_window(
                telemetry,
                duration_seconds=args.duration_seconds,
                interval_seconds=args.interval_seconds,
                npu_smi_path=args.npu_smi,
            )
        report = assess_host_telemetry_file(
            telemetry,
            target_npu_id=args.target_npu,
            policy=policy,
        )
    else:
        policy = CampaignInterferencePolicy(
            max_host_cpu_busy_p95_range=args.max_host_cpu_busy_p95_range,
            max_host_iowait_p95_range=args.max_host_iowait_p95_range,
            max_run_queue_per_cpu_p95_range=args.max_run_queue_per_cpu_p95_range,
            max_memory_available_fraction_range=args.max_memory_available_fraction_range,
            max_sibling_aicore_mean_range_percent=(
                args.max_sibling_aicore_mean_range_percent
            ),
            max_target_initial_temperature_range_c=(
                args.max_target_initial_temperature_range_c
            ),
            max_target_initial_power_range_watts=args.max_target_initial_power_range_watts,
        )
        report = assess_campaign_host_report_files(
            args.campaign_dir,
            plan_path=args.plan,
            policy=policy,
        )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["status"])
    return 0 if report["status"] == "clean" else 3


if __name__ == "__main__":
    raise SystemExit(main())
