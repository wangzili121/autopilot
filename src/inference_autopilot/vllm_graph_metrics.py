"""Parse role-delimited vLLM CUDA/ACL graph metrics into audited artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from inference_autopilot.calibration.models import (
    canonical_sha256,
    require_digest,
    require_id,
    require_object,
)


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_LOG_PREFIX = re.compile(
    r"^(?:\([^)]*\)\s+)?(?:DEBUG|INFO|WARNING|ERROR)\s+"
    r"\S+\s+\S+\s+\[[^]]+\]\s*"
)
_BEGIN = re.compile(r"^\[inference-autopilot] graph-stats-begin role=(\S+)$")
_END = re.compile(r"^\[inference-autopilot] graph-stats-end role=(\S+)$")
_MODE = re.compile(r"^- Mode:\s*(\S+)\s*$")
_CAPTURE_SIZES = re.compile(r"^- Capture sizes:\s*(\[.*])\s*$")


def _exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _audit_stat_group(stats: Sequence["VLLMGraphStat"]) -> dict[str, Any]:
    total = sum(stat.count for stat in stats)
    graph = sum(stat.count for stat in stats if stat.runtime_mode != "NONE")
    padded_units = sum(stat.num_paddings * stat.count for stat in stats)
    token_units = sum(stat.num_unpadded_tokens * stat.count for stat in stats)
    bucket_counts: dict[int, int] = defaultdict(int)
    eager_counts: dict[int, int] = defaultdict(int)
    for stat in stats:
        if stat.runtime_mode == "NONE":
            eager_counts[stat.num_unpadded_tokens] += stat.count
        else:
            bucket_counts[stat.num_padded_tokens] += stat.count
    return {
        "event_count": total,
        "graph_event_count": graph,
        "graph_hit_rate": graph / total if total else 0.0,
        "padding_units": padded_units,
        "padding_ratio": padded_units / token_units if token_units else 0.0,
        "bucket_event_counts": {
            str(size): count for size, count in sorted(bucket_counts.items())
        },
        "eager_shape_counts": {
            str(size): count for size, count in sorted(eager_counts.items())
        },
    }


@dataclass(frozen=True, slots=True)
class VLLMGraphStat:
    num_unpadded_tokens: int
    num_padded_tokens: int
    num_paddings: int
    runtime_mode: str
    count: int
    execution_phase: str = "unknown"

    def __post_init__(self) -> None:
        _positive_int(self.num_unpadded_tokens, "num_unpadded_tokens")
        _positive_int(self.num_padded_tokens, "num_padded_tokens")
        _nonnegative_int(self.num_paddings, "num_paddings")
        _positive_int(self.count, "graph stat count")
        if not self.runtime_mode or not self.runtime_mode.replace("_", "").isalnum():
            raise ValueError(f"invalid graph runtime mode: {self.runtime_mode}")
        if self.execution_phase not in {"unknown", "prefill", "decode", "mixed"}:
            raise ValueError(f"invalid graph execution phase: {self.execution_phase}")
        if self.num_padded_tokens - self.num_unpadded_tokens != self.num_paddings:
            raise ValueError("graph stat padding fields are inconsistent")
        if self.runtime_mode == "NONE" and self.num_paddings != 0:
            raise ValueError("eager graph stat cannot contain padding")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_unpadded_tokens": self.num_unpadded_tokens,
            "num_padded_tokens": self.num_padded_tokens,
            "num_paddings": self.num_paddings,
            "runtime_mode": self.runtime_mode,
            "execution_phase": self.execution_phase,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "VLLMGraphStat":
        expected = {
            "num_unpadded_tokens",
            "num_padded_tokens",
            "num_paddings",
            "runtime_mode",
            "count",
        }
        if "execution_phase" in raw:
            expected.add("execution_phase")
        _exact_keys(raw, expected, "vLLM graph stat")
        return cls(
            num_unpadded_tokens=_positive_int(
                raw["num_unpadded_tokens"], "num_unpadded_tokens"
            ),
            num_padded_tokens=_positive_int(
                raw["num_padded_tokens"], "num_padded_tokens"
            ),
            num_paddings=_nonnegative_int(raw["num_paddings"], "num_paddings"),
            runtime_mode=str(raw["runtime_mode"]),
            count=_positive_int(raw["count"], "graph stat count"),
            execution_phase=str(raw.get("execution_phase", "unknown")),
        )


@dataclass(frozen=True, slots=True)
class VLLMEngineGraphProfile:
    engine_role: str
    graph_mode: str
    configured_capture_sizes: tuple[int, ...]
    stats: tuple[VLLMGraphStat, ...]

    def __post_init__(self) -> None:
        if self.engine_role not in {"base", "proposal"}:
            raise ValueError(f"unsupported vLLM engine role: {self.engine_role}")
        if not self.graph_mode:
            raise ValueError("vLLM graph mode cannot be empty")
        if not self.graph_mode.replace("_", "").isalnum():
            raise ValueError(f"invalid vLLM graph mode: {self.graph_mode}")
        if self.graph_mode == "NONE" and self.configured_capture_sizes:
            raise ValueError("vLLM no-graph profile cannot have capture sizes")
        if self.graph_mode != "NONE" and not self.configured_capture_sizes:
            raise ValueError("enabled vLLM graph profile requires capture sizes")
        if tuple(sorted(set(self.configured_capture_sizes))) != self.configured_capture_sizes:
            raise ValueError("configured capture sizes must be unique and sorted")
        for size in self.configured_capture_sizes:
            _positive_int(size, "configured capture size")
        if not self.stats:
            raise ValueError("vLLM graph profile requires runtime statistics")
        keys = [
            (
                stat.num_unpadded_tokens,
                stat.num_padded_tokens,
                stat.runtime_mode,
                stat.execution_phase,
            )
            for stat in self.stats
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "vLLM graph statistics must be aggregated by shape, mode and phase"
            )
        configured = set(self.configured_capture_sizes)
        unknown_buckets = sorted(
            {
                stat.num_padded_tokens
                for stat in self.stats
                if stat.runtime_mode != "NONE"
                and stat.num_padded_tokens not in configured
            }
        )
        if unknown_buckets:
            raise ValueError(
                f"graph runtime used unconfigured capture sizes: {unknown_buckets}"
            )
        if self.graph_mode == "NONE" and any(
            stat.runtime_mode != "NONE" for stat in self.stats
        ):
            raise ValueError("vLLM no-graph profile contains graph runtime events")

    @property
    def used_capture_sizes(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    stat.num_padded_tokens
                    for stat in self.stats
                    if stat.runtime_mode != "NONE"
                }
            )
        )

    @property
    def unused_capture_sizes(self) -> tuple[int, ...]:
        used = set(self.used_capture_sizes)
        return tuple(size for size in self.configured_capture_sizes if size not in used)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_role": self.engine_role,
            "graph_mode": self.graph_mode,
            "configured_capture_sizes": list(self.configured_capture_sizes),
            "stats": [stat.to_dict() for stat in self.stats],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "VLLMEngineGraphProfile":
        _exact_keys(
            raw,
            {"engine_role", "graph_mode", "configured_capture_sizes", "stats"},
            "vLLM engine graph profile",
        )
        sizes = raw["configured_capture_sizes"]
        stats = raw["stats"]
        if not isinstance(sizes, list):
            raise ValueError("configured_capture_sizes must be an array")
        if not isinstance(stats, list):
            raise ValueError("vLLM graph stats must be an array")
        return cls(
            engine_role=str(raw["engine_role"]),
            graph_mode=str(raw["graph_mode"]),
            configured_capture_sizes=tuple(
                _positive_int(size, "configured capture size") for size in sizes
            ),
            stats=tuple(
                VLLMGraphStat.from_dict(require_object(item, "vLLM graph stat"))
                for item in stats
            ),
        )

    def audit(self) -> dict[str, Any]:
        aggregate = _audit_stat_group(self.stats)
        total = aggregate["event_count"]
        phase_metrics = {
            phase: _audit_stat_group(
                tuple(stat for stat in self.stats if stat.execution_phase == phase)
            )
            for phase in sorted({stat.execution_phase for stat in self.stats})
        }
        return {
            "engine_role": self.engine_role,
            "graph_mode": self.graph_mode,
            **aggregate,
            "configured_capture_sizes": list(self.configured_capture_sizes),
            "used_capture_sizes": list(self.used_capture_sizes),
            "trace_preserving_capture_sizes": list(self.used_capture_sizes),
            "unused_capture_sizes": list(self.unused_capture_sizes),
            "phase_coverage_rate": (
                1.0
                - sum(
                    stat.count
                    for stat in self.stats
                    if stat.execution_phase == "unknown"
                )
                / total
            ),
            "phase_metrics": phase_metrics,
        }

    def diagnose(self) -> dict[str, Any]:
        """Classify graph misses by execution mechanism, without timing claims."""

        audit = self.audit()
        total = int(audit["event_count"])
        if audit["phase_coverage_rate"] < 1.0:
            return {
                "engine_role": self.engine_role,
                "graph_mode": self.graph_mode,
                "status": "insufficient_phase_evidence",
                "phase_coverage_rate": audit["phase_coverage_rate"],
                "interventions": [],
            }
        if self.graph_mode == "NONE":
            return {
                "engine_role": self.engine_role,
                "graph_mode": self.graph_mode,
                "status": "graph_disabled",
                "phase_coverage_rate": 1.0,
                "interventions": [],
            }
        if self.graph_mode != "FULL_DECODE_ONLY":
            return {
                "engine_role": self.engine_role,
                "graph_mode": self.graph_mode,
                "status": "unsupported_graph_mode",
                "phase_coverage_rate": 1.0,
                "interventions": [],
            }

        ceiling = max(self.configured_capture_sizes)
        decode_stats = tuple(
            stat for stat in self.stats if stat.execution_phase == "decode"
        )
        eager_decode = tuple(
            stat for stat in decode_stats if stat.runtime_mode == "NONE"
        )
        above_ceiling = sum(
            stat.count for stat in eager_decode if stat.num_unpadded_tokens > ceiling
        )
        within_ceiling = sum(
            stat.count for stat in eager_decode if stat.num_unpadded_tokens <= ceiling
        )
        mixed = sum(
            stat.count for stat in self.stats if stat.execution_phase == "mixed"
        )
        prefill = sum(
            stat.count for stat in self.stats if stat.execution_phase == "prefill"
        )
        decode_events = sum(stat.count for stat in decode_stats)
        decode_graph_events = sum(
            stat.count for stat in decode_stats if stat.runtime_mode != "NONE"
        )
        interventions = []
        if above_ceiling:
            interventions.append(
                {
                    "mechanism": "decode_above_capture_ceiling",
                    "event_count": above_ceiling,
                    "event_rate": above_ceiling / total,
                    "candidate_knobs": [
                        "max_cudagraph_capture_size",
                        "cudagraph_capture_sizes",
                        "max_num_seqs",
                        "algorithm_admission_limit",
                    ],
                }
            )
        if within_ceiling:
            interventions.append(
                {
                    "mechanism": "decode_within_ceiling_eager",
                    "event_count": within_ceiling,
                    "event_rate": within_ceiling / total,
                    "candidate_knobs": [
                        "cudagraph_capture_sizes",
                        "graph_compatibility_constraints",
                    ],
                }
            )
        if mixed:
            interventions.append(
                {
                    "mechanism": "prefill_decode_mixing",
                    "event_count": mixed,
                    "event_rate": mixed / total,
                    "candidate_knobs": [
                        "algorithm_stage_admission",
                        "prefill_decode_scheduling",
                        "chunked_prefill_policy",
                    ],
                }
            )
        interventions.sort(
            key=lambda item: (-int(item["event_count"]), str(item["mechanism"]))
        )
        return {
            "engine_role": self.engine_role,
            "graph_mode": self.graph_mode,
            "status": "diagnosed",
            "phase_coverage_rate": 1.0,
            "event_count": total,
            "decode_event_count": decode_events,
            "decode_graph_event_count": decode_graph_events,
            "decode_graph_hit_rate": (
                decode_graph_events / decode_events if decode_events else 0.0
            ),
            "capture_ceiling": ceiling,
            "prefill_event_count": prefill,
            "mixed_event_count": mixed,
            "interventions": interventions,
        }


@dataclass(frozen=True, slots=True)
class VLLMGraphMetricsProfile:
    profile_id: str
    source_log_sha256: str
    engines: tuple[VLLMEngineGraphProfile, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported vLLM graph metrics profile: {self.schema_version}")
        require_id(self.profile_id, "vLLM graph metrics profile_id")
        require_digest(self.source_log_sha256, "source log SHA256")
        roles = [engine.engine_role for engine in self.engines]
        if not roles or len(roles) != len(set(roles)):
            raise ValueError("vLLM graph metrics profile requires unique engine roles")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "source_log_sha256": self.source_log_sha256,
            "engines": [
                engine.to_dict()
                for engine in sorted(self.engines, key=lambda item: item.engine_role)
            ],
        }

    def artifact_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {**payload, "vllm_graph_metrics_sha256": canonical_sha256(payload)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "VLLMGraphMetricsProfile":
        _exact_keys(
            raw,
            {
                "schema_version",
                "profile_id",
                "source_log_sha256",
                "engines",
                "vllm_graph_metrics_sha256",
            },
            "vLLM graph metrics profile",
        )
        payload = dict(raw)
        digest = str(payload.pop("vllm_graph_metrics_sha256"))
        require_digest(digest, "vLLM graph metrics SHA256")
        if canonical_sha256(payload) != digest:
            raise ValueError("vLLM graph metrics SHA256 does not match its content")
        engines = payload["engines"]
        if not isinstance(engines, list):
            raise ValueError("vLLM graph metrics engines must be an array")
        return cls(
            schema_version=str(payload["schema_version"]),
            profile_id=str(payload["profile_id"]),
            source_log_sha256=str(payload["source_log_sha256"]),
            engines=tuple(
                VLLMEngineGraphProfile.from_dict(
                    require_object(item, "vLLM engine graph profile")
                )
                for item in engines
            ),
        )

    def audit(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "engines": [
                engine.audit()
                for engine in sorted(self.engines, key=lambda item: item.engine_role)
            ],
        }

    def diagnose(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "engines": [
                engine.diagnose()
                for engine in sorted(self.engines, key=lambda item: item.engine_role)
            ],
        }


def _payload(line: str) -> str:
    clean = _ANSI.sub("", line).replace("\r", "").strip()
    return _LOG_PREFIX.sub("", clean)


def parse_vllm_graph_metrics(
    log_text: str,
    *,
    profile_id: str,
    source_log_sha256: str | None = None,
) -> VLLMGraphMetricsProfile:
    """Parse graph tables emitted between Autopilot role delimiters."""

    require_id(profile_id, "vLLM graph metrics profile_id")
    if source_log_sha256 is None:
        source_log_sha256 = hashlib.sha256(log_text.encode("utf-8")).hexdigest()
    require_digest(source_log_sha256, "source log SHA256")
    current_role: str | None = None
    builders: dict[str, dict[str, Any]] = {}
    for raw_line in log_text.splitlines():
        line = _payload(raw_line)
        begin = _BEGIN.fullmatch(line)
        if begin:
            role = begin.group(1)
            if current_role is not None:
                raise ValueError("nested graph metrics role delimiters")
            if role in builders:
                raise ValueError(f"duplicate graph metrics block for role: {role}")
            current_role = role
            builders[role] = {"mode": None, "sizes": None, "stats": defaultdict(int)}
            continue
        end = _END.fullmatch(line)
        if end:
            if current_role != end.group(1):
                raise ValueError("mismatched graph metrics role delimiter")
            current_role = None
            continue
        if current_role is None:
            continue

        builder = builders[current_role]
        mode = _MODE.fullmatch(line)
        if mode:
            builder["mode"] = mode.group(1)
            continue
        capture_sizes = _CAPTURE_SIZES.fullmatch(line)
        if capture_sizes:
            parsed_sizes = json.loads(capture_sizes.group(1))
            if not isinstance(parsed_sizes, list):
                raise ValueError("vLLM capture sizes must be an array")
            builder["sizes"] = tuple(
                sorted(_positive_int(value, "vLLM capture size") for value in parsed_sizes)
            )
            continue
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) not in {5, 6} or not columns[0].isdigit():
            continue
        if len(columns) == 5:
            unpadded, padded, paddings, runtime_mode, count = columns
            execution_phase = "unknown"
        else:
            unpadded, padded, paddings, runtime_mode, execution_phase, count = columns
        key = (
            int(unpadded),
            int(padded),
            int(paddings),
            runtime_mode,
            execution_phase,
        )
        builder["stats"][key] += int(count)

    if current_role is not None:
        raise ValueError(f"unterminated graph metrics block for role: {current_role}")
    if not builders:
        raise ValueError("no role-delimited vLLM graph metrics found")

    engines = []
    for role, builder in sorted(builders.items()):
        if builder["mode"] is None or builder["sizes"] is None or not builder["stats"]:
            raise ValueError(f"incomplete vLLM graph metrics block for role: {role}")
        stats = tuple(
            VLLMGraphStat(
                num_unpadded_tokens=key[0],
                num_padded_tokens=key[1],
                num_paddings=key[2],
                runtime_mode=key[3],
                execution_phase=key[4],
                count=count,
            )
            for key, count in sorted(builder["stats"].items())
        )
        engines.append(
            VLLMEngineGraphProfile(
                engine_role=role,
                graph_mode=builder["mode"],
                configured_capture_sizes=builder["sizes"],
                stats=stats,
            )
        )
    return VLLMGraphMetricsProfile(
        profile_id=profile_id,
        source_log_sha256=source_log_sha256,
        engines=tuple(engines),
    )


def merge_vllm_graph_metrics(
    profiles: Sequence[VLLMGraphMetricsProfile],
    *,
    profile_id: str,
) -> VLLMGraphMetricsProfile:
    """Aggregate compatible graph-shape profiles without losing event counts."""

    require_id(profile_id, "merged vLLM graph metrics profile_id")
    if len(profiles) < 2:
        raise ValueError("merged vLLM graph metrics require at least two profiles")
    digests = [canonical_sha256(profile.to_dict()) for profile in profiles]
    if len(digests) != len(set(digests)):
        raise ValueError("cannot merge a vLLM graph metrics profile more than once")

    reference = {
        engine.engine_role: (
            engine.graph_mode,
            engine.configured_capture_sizes,
        )
        for engine in profiles[0].engines
    }
    aggregates: dict[str, dict[tuple[int, int, int, str, str], int]] = {
        role: defaultdict(int) for role in reference
    }
    for profile in profiles:
        engines = {engine.engine_role: engine for engine in profile.engines}
        actual = {
            role: (engine.graph_mode, engine.configured_capture_sizes)
            for role, engine in engines.items()
        }
        if actual != reference:
            raise ValueError(
                "merged vLLM graph metrics must use identical roles, modes and "
                "configured capture sizes"
            )
        for role, engine in engines.items():
            for stat in engine.stats:
                key = (
                    stat.num_unpadded_tokens,
                    stat.num_padded_tokens,
                    stat.num_paddings,
                    stat.runtime_mode,
                    stat.execution_phase,
                )
                aggregates[role][key] += stat.count

    engines = tuple(
        VLLMEngineGraphProfile(
            engine_role=role,
            graph_mode=reference[role][0],
            configured_capture_sizes=reference[role][1],
            stats=tuple(
                VLLMGraphStat(
                    num_unpadded_tokens=key[0],
                    num_padded_tokens=key[1],
                    num_paddings=key[2],
                    runtime_mode=key[3],
                    execution_phase=key[4],
                    count=count,
                )
                for key, count in sorted(aggregates[role].items())
            ),
        )
        for role in sorted(reference)
    )
    source_digest = canonical_sha256(
        {
            "artifact_type": "merged_vllm_graph_metrics_sources",
            "profile_sha256s": sorted(digests),
        }
    )
    return VLLMGraphMetricsProfile(
        profile_id=profile_id,
        source_log_sha256=source_digest,
        engines=engines,
    )
