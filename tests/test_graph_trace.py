from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from inference_autopilot.cli import main
from inference_autopilot.graph_trace import (
    GraphWorkloadTrace,
    trace_from_chang_result,
)


def _event(role: str, kind: str, shape: int, start: float) -> dict[str, object]:
    duration = shape / 100.0
    return {
        "role": role,
        "kind": kind,
        "start_unix": start,
        "end_unix": start + duration,
        "duration_seconds": duration,
        "request_groups": shape,
    }


def _result() -> dict[str, object]:
    base_shapes = [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    events = [
        _event("base", "sample", shape, 100.0 + index)
        for index, shape in enumerate(base_shapes)
    ]
    events.extend(
        [
            _event("base", "score", 8, 120.0),
            _event("proposal", "sample", 24, 121.0),
        ]
    )
    return {
        "requests": 4,
        "workers": 4,
        "arrival_qps": 0.0,
        "runtime": {
            "base_max_num_seqs": 40,
            "proposal_max_num_seqs": 96,
        },
        "algorithm": {"rollout_count": 3},
        "model_call_timeline": list(reversed(events)),
    }


class GraphTraceTest(unittest.TestCase):
    def test_extracts_semantic_stages_and_recommends_observed_sizes(self) -> None:
        trace = trace_from_chang_result(_result(), trace_id="trace-1")
        audit = trace.audit()

        self.assertEqual(trace.events[0].started_at_seconds, 100.0)
        self.assertEqual(
            {event.stage_id for event in trace.events},
            {"candidate_generate", "target_score", "proposal_rollout_generate"},
        )
        self.assertEqual(
            trace.recommend_request_group_boundaries(
                "base", maximum_candidates=4
            ),
            (1, 2, 8, 10),
        )
        self.assertEqual(
            audit["request_group_boundaries_by_stage"]["candidate_generate"],
            [1, 2, 3, 4, 5, 7, 9, 10],
        )
        self.assertEqual(audit["event_count"], 13)
        self.assertEqual(
            trace.metadata["measurement_semantics"],
            "backend_call_wall_service_time_including_queueing",
        )

    def test_artifact_round_trip_and_digest_tamper_detection(self) -> None:
        trace = trace_from_chang_result(_result(), trace_id="trace-1")
        artifact = trace.artifact_dict()
        restored = GraphWorkloadTrace.from_dict(artifact)
        self.assertEqual(restored.artifact_dict(), artifact)

        tampered = deepcopy(artifact)
        tampered["events"][0]["shape_size"] = 99
        with self.assertRaisesRegex(ValueError, "SHA256"):
            GraphWorkloadTrace.from_dict(tampered)

    def test_rejects_inconsistent_duration_and_unknown_call(self) -> None:
        raw = _result()
        raw["model_call_timeline"][0]["duration_seconds"] = 99.0
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            trace_from_chang_result(raw, trace_id="trace-1")

        raw = _result()
        raw["model_call_timeline"][0]["kind"] = "embed"
        with self.assertRaisesRegex(ValueError, "unsupported chang model call"):
            trace_from_chang_result(raw, trace_id="trace-1")

    def test_cli_extracts_and_audits_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            trace_path = root / "trace.json"
            result_path.write_text(json.dumps(_result()), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "trace-graph-workload",
                            str(result_path),
                            "--trace-id",
                            "trace-cli",
                            "--output",
                            str(trace_path),
                        ]
                    ),
                    0,
                )
                self.assertEqual(main(["audit-graph-trace", str(trace_path)]), 0)
            artifact = json.loads(trace_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact["trace_id"], "trace-cli")
        self.assertIn("graph_workload_trace_sha256", artifact)


if __name__ == "__main__":
    unittest.main()
