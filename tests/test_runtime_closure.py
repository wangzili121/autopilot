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
from inference_autopilot.calibration.models import RunObservation, canonical_sha256
from inference_autopilot.features import SelectorFeatureTable, features_from_run
from inference_autopilot.runtime_closure import (
    RuntimeClosureAssessment,
    attest_runtime_closure,
    enrich_features_with_runtime_closure,
)


ROOT = Path(__file__).resolve().parents[1]


def _capture_sizes(capacity: int) -> list[int]:
    ceiling = min(capacity, 512)
    return [1, ceiling] if ceiling > 1 else [1]


def _engine_log(
    role: str,
    settings: dict[str, object],
    *,
    force_capture_sizes: list[int] | None = None,
) -> str:
    capacity = int(settings[f"{role}_max_num_seqs"])
    token_budget = int(settings[f"{role}_max_num_batched_tokens"])
    capture_sizes = force_capture_sizes or _capture_sizes(capacity)
    graph_mode = str(settings.get(f"{role}_graph_mode", "FULL_DECODE_ONLY"))
    model = "/models/base" if role == "base" else "/models/proposal"
    config = (
        "Initializing a V1 LLM engine (v0.18.0) with config: "
        f"model='{model}', dtype=torch.bfloat16, max_seq_len=32768, "
        "enable_prefix_caching=True, enable_chunked_prefill=True, "
        "compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, "
        "'backend': 'vllm_ascend.compilation.compiler_interface.AscendCompiler', "
        f"'compile_ranges_endpoints': [{token_budget}], "
        f"'cudagraph_mode': <CUDAGraphMode.{graph_mode}: (2, 0)>, "
        f"'cudagraph_capture_sizes': {capture_sizes}, "
        f"'max_cudagraph_capture_size': {max(capture_sizes)}}}"
    )
    return "\n".join(
        (
            config,
            "Available KV cache memory: 18.25 GiB",
            "GPU KV cache size: 1,234,560 tokens",
            "Maximum concurrency for 32,768 tokens per request: 37.67x",
            "Graph capturing finished in 7 secs, took 0.33 GiB",
        )
    )


class RuntimeClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = json.loads(
            (
                ROOT
                / "examples/conditional-is-short-p96.calibration.example.json"
            ).read_text(encoding="utf-8")
        )
        self.plan = build_plan(CalibrationSpec.from_dict(raw))

    def _write_campaign(
        self,
        root: Path,
        *,
        missing_log_run: str | None = None,
        forced_capture_run: str | None = None,
    ) -> None:
        for expected in self.plan.runs:
            run_root = root / expected.run_id
            run_root.mkdir(parents=True)
            manifest = build_run_manifest(self.plan, expected.run_id)
            settings = dict(manifest["configuration"]["settings"])
            (run_root / "run-manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            effective = {
                "deployment_settings": settings,
                "semantic_invariants": manifest["semantic_contract"]["invariants"],
                "source_config_sha256": "a" * 64,
            }
            (run_root / "effective-config.json").write_text(
                json.dumps(effective), encoding="utf-8"
            )
            if expected.run_id == missing_log_run:
                continue
            forced = [1, 8] if expected.run_id == forced_capture_run else None
            (run_root / "runner.log").write_text(
                "\n".join(
                    (
                        _engine_log("base", settings, force_capture_sizes=forced),
                        _engine_log("proposal", settings),
                    )
                ),
                encoding="utf-8",
            )

    def test_attests_runtime_resolved_graph_policy(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root)
            assessment = attest_runtime_closure(self.plan, root)

        self.assertTrue(assessment.complete)
        self.assertEqual(assessment.valid_run_count, len(self.plan.runs))
        self.assertEqual(
            assessment.summary["runtime_resolved_graph_policy_engine_count"],
            len(self.plan.runs) * 2,
        )
        self.assertGreater(
            assessment.summary["graph_capacity_gap_engine_count"], 0
        )
        self.assertEqual(
            RuntimeClosureAssessment.from_dict(assessment.to_dict()), assessment
        )
        self.assertTrue(assessment.audit()["complete"])

    def test_explicit_graph_policy_mismatch_fails_closed(self) -> None:
        raw = json.loads(
            (
                ROOT
                / "examples/conditional-is-short-p96.calibration.example.json"
            ).read_text(encoding="utf-8")
        )
        for configuration in (raw["baseline"], *raw["candidates"]):
            settings = configuration["settings"]
            settings["base_graph_mode"] = "FULL_DECODE_ONLY"
            settings["base_graph_capture_sizes"] = _capture_sizes(
                settings["base_max_num_seqs"]
            )
            settings["proposal_graph_mode"] = "FULL_DECODE_ONLY"
            settings["proposal_graph_capture_sizes"] = _capture_sizes(
                settings["proposal_max_num_seqs"]
            )
        self.plan = build_plan(CalibrationSpec.from_dict(raw))
        corrupt = self.plan.runs[0].run_id
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root, forced_capture_run=corrupt)
            assessment = attest_runtime_closure(self.plan, root)

        self.assertFalse(assessment.complete)
        self.assertEqual(assessment.valid_run_count, len(self.plan.runs))
        self.assertEqual(assessment.summary["graph_policy_mismatch_count"], 1)
        self.assertIn("did not honor requested settings", assessment.issues[0].message)

    def test_wavefront_capacity_closes_only_the_adapter_effective_domain(self) -> None:
        raw = json.loads(
            (
                ROOT
                / "examples/npu/conditional-is-short-p96-stage-wavefront-abba-20260906.calibration.json"
            ).read_text(encoding="utf-8")
        )
        self.plan = build_plan(CalibrationSpec.from_dict(raw))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root)
            assessment = attest_runtime_closure(self.plan, root)

        candidate_proposal = next(
            engine
            for run in assessment.runs
            if run.configuration_id == "stage-wavefront-auto-2d"
            for engine in run.engines
            if engine.role == "proposal"
        )
        control_proposal = next(
            engine
            for run in assessment.runs
            if run.configuration_id == "stage-wavefront-off"
            for engine in run.engines
            if engine.role == "proposal"
        )
        self.assertEqual(candidate_proposal.requested_max_num_seqs, 768)
        self.assertEqual(candidate_proposal.effective_admission_max_num_seqs, 384)
        self.assertEqual(
            candidate_proposal.admission_capacity_source,
            "stage_wavefront_plan",
        )
        self.assertEqual(candidate_proposal.uncovered_graph_capacity, 0)
        self.assertEqual(candidate_proposal.graph_capacity_coverage_ratio, 1.0)
        self.assertEqual(control_proposal.effective_admission_max_num_seqs, 768)
        self.assertEqual(control_proposal.uncovered_graph_capacity, 256)
        self.assertEqual(
            assessment.summary["graph_capacity_gap_engine_count"], 2
        )

    def test_prefix_sharding_closes_the_atomic_prefix_run_domain(self) -> None:
        raw = json.loads(
            (
                ROOT
                / "examples/npu/conditional-is-short-p96-stage-wavefront-abba-20260906.calibration.json"
            ).read_text(encoding="utf-8")
        )
        candidate = raw["candidates"][0]
        raw["semantic_contract"]["invariants"]["candidate_count"] = 15
        candidate["settings"]["proposal_stage_wavefront_mode"] = (
            "prefix_sharded"
        )
        candidate["settings"][
            "proposal_stage_wavefront_max_shards_per_parent"
        ] = 1
        candidate["settings"][
            "proposal_stage_wavefront_max_shard_sequences"
        ] = 3
        candidate["settings"][
            "proposal_stage_wavefront_max_shard_prefill_tokens"
        ] = 8192
        candidate["settings"][
            "proposal_stage_wavefront_max_inflight_waves"
        ] = 16
        self.plan = build_plan(CalibrationSpec.from_dict(raw))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root)
            assessment = attest_runtime_closure(self.plan, root)

        candidate_proposal = next(
            engine
            for run in assessment.runs
            if run.configuration_id == "stage-wavefront-auto-2d"
            for engine in run.engines
            if engine.role == "proposal"
        )
        self.assertEqual(candidate_proposal.requested_max_num_seqs, 768)
        self.assertEqual(candidate_proposal.effective_admission_max_num_seqs, 480)
        self.assertEqual(
            candidate_proposal.admission_capacity_source,
            "stage_wavefront_plan",
        )

    def test_enriches_only_exactly_matched_feature_rows(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root)
            assessment = attest_runtime_closure(self.plan, root)
            rows = []
            for expected in self.plan.runs:
                manifest = build_run_manifest(self.plan, expected.run_id)
                observation = RunObservation(
                    run_id=expected.run_id,
                    run_manifest_sha256=manifest["run_manifest_sha256"],
                    started_at_unix=1.0,
                    finished_at_unix=2.0,
                    status="success",
                    metrics={
                        "completed_qps": 1.0,
                        "latency_seconds": {"p95": 1.0},
                    },
                )
                rows.extend(features_from_run(manifest, observation).rows)
            table = SelectorFeatureTable(tuple(rows))
            enriched = enrich_features_with_runtime_closure(table, assessment)

        self.assertEqual(len(enriched.rows), len(table.rows))
        for row in enriched.rows:
            self.assertIn("runtime.base.graph_capture_ceiling", row.static_features)
            self.assertIn(
                "runtime_closure.proposal.available_kv_cache_gib",
                row.telemetry_features,
            )
            self.assertIn("runtime_closure_attested", row.tags)

        with self.assertRaisesRegex(ValueError, "do not match exactly"):
            enrich_features_with_runtime_closure(
                SelectorFeatureTable(table.rows[:-1]), assessment
            )

    def test_rehashed_summary_tampering_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root)
            payload = attest_runtime_closure(self.plan, root).to_dict()

        payload["summary"]["graph_capacity_gap_engine_count"] += 1
        unhashed = dict(payload)
        unhashed.pop("runtime_closure_assessment_sha256")
        payload["runtime_closure_assessment_sha256"] = canonical_sha256(unhashed)
        with self.assertRaisesRegex(ValueError, "summary does not match"):
            RuntimeClosureAssessment.from_dict(payload)

    def test_missing_log_is_reported_without_dropping_other_runs(self) -> None:
        missing = self.plan.runs[-1].run_id
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root, missing_log_run=missing)
            assessment = attest_runtime_closure(self.plan, root)

        self.assertFalse(assessment.complete)
        self.assertEqual(assessment.valid_run_count, len(self.plan.runs) - 1)
        self.assertIn("missing runtime closure artifact", assessment.issues[0].message)


if __name__ == "__main__":
    unittest.main()
