from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from inference_autopilot.calibration import (
    CalibrationSpec,
    build_plan,
    build_run_manifest,
)
from inference_autopilot.calibration.models import canonical_sha256
from inference_autopilot.harness_cost import (
    HarnessCostAssessment,
    assess_harness_cost,
)


ROOT = Path(__file__).resolve().parents[1]


def _runner_log(index: int) -> str:
    base_init = 50.0 + index
    proposal_init = 30.0 + index
    return "\n".join(
        (
            "Initializing a V1 LLM engine (v0.18.0) with config: "
            "model='/models/base'",
            "Loading weights took 0.80 seconds",
            "Using cache directory: "
            "/root/.cache/vllm/torch_compile_cache/aaa111/rank_0_0/backbone",
            "Dynamo bytecode transform time: 5.50 s",
            "Compiling a graph for compile range (1, 10240) takes 12.00 s",
            "torch.compile and initial profiling/warmup run together took "
            "45.00 s in total",
            "Graph capturing finished in 8 secs, took 0.33 GiB",
            "init engine (profile, create kv cache, warmup model) took "
            f"{base_init:.2f} seconds",
            "Initializing a V1 LLM engine (v0.18.0) with config: "
            "model='/models/proposal'",
            "Loading weights took 0.35 seconds",
            "Using cache directory: "
            "/root/.cache/vllm/torch_compile_cache/bbb222/rank_0_0/backbone",
            "Dynamo bytecode transform time: 4.80 s",
            "Compiling a graph for compile range (1, 12288) takes 9.50 s",
            "torch.compile and initial profiling/warmup run together took "
            "20.50 s in total",
            "Graph capturing finished in 7 secs, took 0.32 GiB",
            "init engine (profile, create kv cache, warmup model) took "
            f"{proposal_init:.2f} seconds",
        )
    )


class HarnessCostTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = CalibrationSpec.from_dict(
            json.loads(
                (ROOT / "examples/conditional-is-short-p96.calibration.example.json")
                .read_text(encoding="utf-8")
            )
        )
        self.plan = build_plan(spec)

    def _write_campaign(self, root: Path, *, corrupt_run: str | None = None) -> None:
        for expected in self.plan.runs:
            run_root = root / expected.run_id
            run_root.mkdir(parents=True)
            manifest_payload = build_run_manifest(self.plan, expected.run_id)
            (run_root / "run-manifest.json").write_text(
                json.dumps(manifest_payload), encoding="utf-8"
            )
            settings = dict(manifest_payload["configuration"]["settings"])
            if expected.run_id == corrupt_run:
                settings["base_max_num_seqs"] = 999
            effective = {
                "deployment_settings": settings,
                "semantic_invariants": manifest_payload["semantic_contract"][
                    "invariants"
                ],
                "source_config_sha256": "a" * 64,
            }
            (run_root / "effective-config.json").write_text(
                json.dumps(effective), encoding="utf-8"
            )
            (run_root / "runner.log").write_text(
                _runner_log(expected.sequence_index), encoding="utf-8"
            )

    def test_extracts_repeated_compile_and_graph_costs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root)
            assessment = assess_harness_cost(self.plan, root)

        self.assertTrue(assessment.complete)
        self.assertEqual(assessment.valid_run_count, len(self.plan.runs))
        self.assertGreater(
            assessment.summary["repeated_compiler_invocation_count"], 0
        )
        self.assertGreater(assessment.summary["repeated_graph_capture_count"], 0)
        self.assertTrue(
            all(
                group.classification == "cache_key_reused_compile_repeated"
                for group in assessment.cache_reuse_groups
                if len(group.run_ids) > 1
            )
        )
        self.assertEqual(
            HarnessCostAssessment.from_dict(assessment.to_dict()), assessment
        )
        self.assertTrue(assessment.audit()["complete"])

    def test_effective_configuration_mismatch_is_an_issue(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            corrupt = self.plan.runs[0].run_id
            self._write_campaign(root, corrupt_run=corrupt)
            assessment = assess_harness_cost(self.plan, root)

        self.assertFalse(assessment.complete)
        self.assertEqual(assessment.valid_run_count, len(self.plan.runs) - 1)
        self.assertEqual(assessment.issues[0].run_id, corrupt)

    def test_rehashed_derived_summary_tampering_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root)
            payload = assess_harness_cost(self.plan, root).to_dict()

        payload["summary"]["total_engine_startup_seconds"] += 1.0
        unhashed = dict(payload)
        unhashed.pop("harness_cost_assessment_sha256")
        payload["harness_cost_assessment_sha256"] = canonical_sha256(unhashed)
        with self.assertRaisesRegex(ValueError, "summary does not match"):
            HarnessCostAssessment.from_dict(payload)

    def test_missing_log_is_reported_without_dropping_other_runs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root)
            missing = self.plan.runs[-1].run_id
            (root / missing / "runner.log").unlink()
            assessment = assess_harness_cost(self.plan, root)

        self.assertFalse(assessment.complete)
        self.assertEqual(assessment.valid_run_count, len(self.plan.runs) - 1)
        self.assertIn("missing harness artifact", assessment.issues[0].message)


if __name__ == "__main__":
    unittest.main()
