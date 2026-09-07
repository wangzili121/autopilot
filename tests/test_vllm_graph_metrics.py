from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from inference_autopilot.cli import main
from inference_autopilot.vllm_graph_metrics import (
    VLLMGraphMetricsProfile,
    merge_vllm_graph_metrics,
    parse_vllm_graph_metrics,
)


_LOG = """
unrelated initialization output
[inference-autopilot] graph-stats-begin role=proposal
INFO 09-04 09:32:40 [cuda_graph.py:123] **CUDAGraph Config Settings:**
INFO 09-04 09:32:40 [cuda_graph.py:123] - Mode: FULL_DECODE_ONLY
INFO 09-04 09:32:40 [cuda_graph.py:123] - Capture sizes: [1, 2, 4, 8]
INFO 09-04 09:32:40 [cuda_graph.py:123] | Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count |
INFO 09-04 09:32:40 [cuda_graph.py:123] |-----------------|---------------|--------------|--------------|-------|
INFO 09-04 09:32:40 [cuda_graph.py:123] | 7               | 8             | 1            | FULL         | 3     |
INFO 09-04 09:32:40 [cuda_graph.py:123] | 2               | 2             | 0            | FULL         | 2     |
INFO 09-04 09:32:40 [cuda_graph.py:123] | 300             | 300           | 0            | NONE         | 1     |
[inference-autopilot] graph-stats-end role=proposal
[inference-autopilot] graph-stats-begin role=base
(Engine pid=1) INFO 09-04 09:32:40 [cuda_graph.py:123] - Mode: FULL_DECODE_ONLY
(Engine pid=1) INFO 09-04 09:32:40 [cuda_graph.py:123] - Capture sizes: [1, 2, 4]
(Engine pid=1) INFO 09-04 09:32:40 [cuda_graph.py:123] | 1 | 1 | 0 | FULL | 4 |
[inference-autopilot] graph-stats-end role=base
"""

_PHASE_LOG = """
[inference-autopilot] graph-stats-begin role=proposal
INFO 09-04 09:32:40 [cuda_graph.py:123] - Mode: FULL_DECODE_ONLY
INFO 09-04 09:32:40 [cuda_graph.py:123] - Capture sizes: [4, 8]
INFO 09-04 09:32:40 [cuda_graph.py:123] | Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Execution Phase | Count |
INFO 09-04 09:32:40 [cuda_graph.py:123] | 7 | 8 | 1 | FULL | decode | 3 |
INFO 09-04 09:32:40 [cuda_graph.py:123] | 4 | 4 | 0 | FULL | mixed | 2 |
INFO 09-04 09:32:40 [cuda_graph.py:123] | 300 | 300 | 0 | NONE | prefill | 1 |
[inference-autopilot] graph-stats-end role=proposal
"""


class VLLMGraphMetricsTest(unittest.TestCase):
    def test_parser_preserves_roles_and_finds_unused_buckets(self) -> None:
        profile = parse_vllm_graph_metrics(_LOG, profile_id="npu-smoke")
        audits = {
            engine["engine_role"]: engine for engine in profile.audit()["engines"]
        }

        self.assertEqual(audits["proposal"]["used_capture_sizes"], [2, 8])
        self.assertEqual(audits["proposal"]["unused_capture_sizes"], [1, 4])
        self.assertEqual(audits["proposal"]["graph_event_count"], 5)
        self.assertAlmostEqual(audits["proposal"]["graph_hit_rate"], 5 / 6)
        self.assertEqual(audits["proposal"]["padding_units"], 3)
        self.assertEqual(audits["base"]["trace_preserving_capture_sizes"], [1])
        self.assertEqual(audits["proposal"]["phase_coverage_rate"], 0.0)
        self.assertEqual(audits["proposal"]["phase_metrics"]["unknown"]["event_count"], 6)

    def test_parser_preserves_execution_phase_and_audits_each_phase(self) -> None:
        profile = parse_vllm_graph_metrics(_PHASE_LOG, profile_id="phase-aware")
        proposal = profile.engines[0]
        audit = proposal.audit()

        self.assertEqual(audit["phase_coverage_rate"], 1.0)
        self.assertEqual(audit["phase_metrics"]["decode"]["event_count"], 3)
        self.assertEqual(audit["phase_metrics"]["decode"]["graph_hit_rate"], 1.0)
        self.assertEqual(audit["phase_metrics"]["decode"]["padding_units"], 3)
        self.assertEqual(audit["phase_metrics"]["mixed"]["bucket_event_counts"], {"4": 2})
        self.assertEqual(audit["phase_metrics"]["prefill"]["graph_hit_rate"], 0.0)

    def test_phase_diagnosis_separates_mixing_from_decode_capture_gaps(self) -> None:
        profile = parse_vllm_graph_metrics(
            _PHASE_LOG.replace(
                "| 7 | 8 | 1 | FULL | decode | 3 |",
                "| 9 | 9 | 0 | NONE | decode | 3 |",
            ),
            profile_id="phase-diagnosis",
        )
        diagnosis = profile.diagnose()["engines"][0]

        self.assertEqual(diagnosis["status"], "diagnosed")
        self.assertEqual(diagnosis["capture_ceiling"], 8)
        self.assertEqual(diagnosis["decode_graph_hit_rate"], 0.0)
        self.assertEqual(
            [item["mechanism"] for item in diagnosis["interventions"]],
            ["decode_above_capture_ceiling", "prefill_decode_mixing"],
        )

    def test_phase_diagnosis_fails_closed_for_legacy_profile(self) -> None:
        profile = parse_vllm_graph_metrics(_LOG, profile_id="legacy-diagnosis")
        diagnoses = profile.diagnose()["engines"]

        self.assertTrue(
            all(item["status"] == "insufficient_phase_evidence" for item in diagnoses)
        )

    def test_merge_keeps_identical_shapes_from_different_phases_separate(self) -> None:
        first = parse_vllm_graph_metrics(_PHASE_LOG, profile_id="phase-first")
        second = parse_vllm_graph_metrics(
            _PHASE_LOG.replace("| FULL | decode | 3 |", "| FULL | mixed | 3 |"),
            profile_id="phase-second",
        )
        merged = merge_vllm_graph_metrics(
            [first, second], profile_id="phase-merged"
        )
        matching = [
            stat
            for stat in merged.engines[0].stats
            if stat.num_unpadded_tokens == 7
        ]

        self.assertEqual(len(matching), 2)
        self.assertEqual(
            {stat.execution_phase for stat in matching}, {"decode", "mixed"}
        )

    def test_artifact_round_trip_and_tamper_detection(self) -> None:
        artifact = parse_vllm_graph_metrics(
            _LOG, profile_id="npu-smoke"
        ).artifact_dict()
        self.assertEqual(
            VLLMGraphMetricsProfile.from_dict(artifact).artifact_dict(), artifact
        )
        tampered = deepcopy(artifact)
        tampered["engines"][0]["stats"][0]["count"] = 99
        with self.assertRaisesRegex(ValueError, "SHA256"):
            VLLMGraphMetricsProfile.from_dict(tampered)

    def test_rejects_missing_delimiters_and_unconfigured_runtime_bucket(self) -> None:
        with self.assertRaisesRegex(ValueError, "role-delimited"):
            parse_vllm_graph_metrics(
                "INFO graph without role markers", profile_id="npu-smoke"
            )

        invalid = _LOG.replace(
            "| 7               | 8             | 1",
            "| 7               | 16            | 9",
        )
        with self.assertRaisesRegex(ValueError, "unconfigured capture sizes"):
            parse_vllm_graph_metrics(invalid, profile_id="npu-smoke")

    def test_cli_imports_and_audits_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "runner.log"
            profile_path = root / "profile.json"
            log_path.write_text(_LOG, encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "import-vllm-graph-metrics",
                            str(log_path),
                            "--profile-id",
                            "npu-smoke",
                            "--output",
                            str(profile_path),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(["audit-vllm-graph-metrics", str(profile_path)]), 0
                )
                self.assertEqual(
                    main(["diagnose-vllm-graph-runtime", str(profile_path)]), 0
                )
            artifact = json.loads(profile_path.read_text(encoding="utf-8"))

        self.assertEqual(len(artifact["engines"]), 2)
        self.assertIn("vllm_graph_metrics_sha256", artifact)
        self.assertEqual(
            artifact["source_log_sha256"],
            hashlib.sha256(_LOG.encode("utf-8")).hexdigest(),
        )

    def test_no_graph_profile_accepts_empty_capture_sizes(self) -> None:
        log = """
[inference-autopilot] graph-stats-begin role=base
INFO 09-04 09:32:40 [cuda_graph.py:123] - Mode: NONE
INFO 09-04 09:32:40 [cuda_graph.py:123] - Capture sizes: []
INFO 09-04 09:32:40 [cuda_graph.py:123] | 17 | 17 | 0 | NONE | 3 |
[inference-autopilot] graph-stats-end role=base
"""
        profile = parse_vllm_graph_metrics(log, profile_id="no-graph")
        audit = profile.engines[0].audit()
        self.assertEqual(audit["graph_hit_rate"], 0.0)
        self.assertEqual(audit["configured_capture_sizes"], [])

    def test_merge_preserves_shapes_from_multiple_regimes(self) -> None:
        first = parse_vllm_graph_metrics(_LOG, profile_id="short")
        second_log = _LOG.replace(
            "| 7               | 8             | 1            | FULL         | 3",
            "| 3               | 4             | 1            | FULL         | 7",
        )
        second = parse_vllm_graph_metrics(second_log, profile_id="mixed")
        merged = merge_vllm_graph_metrics(
            [first, second],
            profile_id="short-mixed-training",
        )
        reversed_merge = merge_vllm_graph_metrics(
            [second, first],
            profile_id="short-mixed-training",
        )
        proposal = next(
            engine for engine in merged.engines if engine.engine_role == "proposal"
        )

        self.assertEqual(merged.artifact_dict(), reversed_merge.artifact_dict())
        self.assertEqual(proposal.used_capture_sizes, (2, 4, 8))
        self.assertEqual(sum(stat.count for stat in proposal.stats), 16)

    def test_merge_rejects_different_framework_bucket_configs(self) -> None:
        first = parse_vllm_graph_metrics(_LOG, profile_id="short")
        incompatible_log = _LOG.replace(
            "Capture sizes: [1, 2, 4, 8]",
            "Capture sizes: [1, 2, 4, 8, 16]",
            1,
        )
        incompatible = parse_vllm_graph_metrics(
            incompatible_log,
            profile_id="different-default",
        )

        with self.assertRaisesRegex(ValueError, "identical roles, modes"):
            merge_vllm_graph_metrics(
                [first, incompatible],
                profile_id="invalid-merge",
            )


if __name__ == "__main__":
    unittest.main()
