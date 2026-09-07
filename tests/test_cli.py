from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from inference_autopilot.calibration.models import CalibrationPlan, RunObservation
from inference_autopilot.calibration.planning import build_run_manifest
from inference_autopilot.cli import main


class CliTest(unittest.TestCase):
    def test_stage_wavefront_cli_selects_graph_eligible_exact_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage-wavefront.json"
            with redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "plan-stage-wavefront",
                        "--outer-concurrency",
                        "96",
                        "--sequences-per-group",
                        "24",
                        "--graph-capture-ceiling",
                        "512",
                        "--scheduler-sequence-cap",
                        "768",
                        "--output",
                        str(output),
                    ]
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(payload["selected"]["groups_per_wave"], 16)
        self.assertEqual(payload["selected"]["sequences_per_wave"], 384)

    def test_graph_command_writes_valid_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "graph.json"
            status = main(
                [
                    "graph",
                    "--candidate-count",
                    "5",
                    "--rollout-count",
                    "2",
                    "--output",
                    str(output),
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        self.assertEqual(payload["algorithm_id"], "conditional_is_small_proposal")
        self.assertEqual(payload["algorithm_semantics"], "exact")

    def test_calibration_cli_round_trip_produces_formal_evidence(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        spec = project_root / "examples" / "conditional-is-short-p96.calibration.example.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            observations_path = root / "observations"
            assessment_path = root / "assessment.json"
            merged_path = root / "merged.json"
            observations_path.mkdir()

            self.assertEqual(
                main(["plan-calibration", str(spec), "--output", str(plan_path)]),
                0,
            )
            plan = CalibrationPlan.from_dict(
                json.loads(plan_path.read_text(encoding="utf-8"))
            )
            for run in plan.runs:
                manifest = build_run_manifest(plan, run.run_id)
                start = float(run.sequence_index * 10)
                observation = RunObservation(
                    run_id=run.run_id,
                    run_manifest_sha256=manifest["run_manifest_sha256"],
                    started_at_unix=start,
                    finished_at_unix=start + 5.0,
                    status="success",
                    metrics={
                        "completed_qps": 0.8,
                        "latency_seconds": {"p50": 90.0, "p95": 100.0, "p99": 110.0},
                        "accuracy": 0.5,
                        "preemptions": 0,
                        "proposal_kv_peak_fraction": 0.15,
                    },
                )
                (observations_path / f"{run.run_id}.json").write_text(
                    json.dumps(observation.to_dict()), encoding="utf-8"
                )

            status = main(
                [
                    "assess-calibration",
                    str(plan_path),
                    str(observations_path),
                    "--output",
                    str(assessment_path),
                ]
            )
            assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
            self.assertEqual(
                main(
                    [
                        "merge-ledgers",
                        str(assessment_path),
                        str(assessment_path),
                        "--output",
                        str(merged_path),
                    ]
                ),
                0,
            )
            merged = json.loads(merged_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertTrue(assessment["formal_complete"])
        self.assertEqual(assessment["formal_group_count"], 2)
        self.assertEqual(
            {record["quality"]["grade"] for record in assessment["ledger"]["records"]},
            {"A_formal_paired"},
        )
        self.assertEqual(len(merged["records"]), len(assessment["ledger"]["records"]))

    def test_feature_cli_round_trip_recomputes_audit(self) -> None:
        legacy = {
            "schema_version": 1,
            "workload": {
                "method": "conditional_is_small_proposal",
                "requests": 32,
                "workers": 32,
                "arrival_qps": 0.0,
            },
            "runs": [
                {
                    "base_max_num_seqs": 128,
                    "proposal_max_num_seqs": 768,
                    "completed_qps": 0.8,
                    "p95_seconds": 6.0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.json"
            ledger = root / "ledger.json"
            table = root / "features.json"
            source.write_text(json.dumps(legacy), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(["import-results", str(source), "--output", str(ledger)]),
                    0,
                )
                self.assertEqual(
                    main(["features-ledger", str(ledger), "--output", str(table)]),
                    0,
                )
                self.assertEqual(main(["audit-features", str(table)]), 0)
            payload = json.loads(table.read_text(encoding="utf-8"))

        self.assertEqual(payload["audit"]["row_count"], 1)
        self.assertEqual(payload["rows"][0]["evidence"]["grade"], "B_controlled_single")

    def test_npu_cost_model_cli_fits_and_predicts_supported_graph(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        calibration = project_root / "examples/npu-stage-calibration.example.json"
        query = project_root / "examples/npu-graph-cost-query.example.json"
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.json"
            prediction = Path(directory) / "prediction.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "fit-npu-cost-model",
                            str(calibration),
                            "--output",
                            str(model),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "predict-npu-graph-cost",
                            str(model),
                            str(query),
                            "--output",
                            str(prediction),
                        ]
                    ),
                    0,
                )
            model_payload = json.loads(model.read_text(encoding="utf-8"))
            prediction_payload = json.loads(prediction.read_text(encoding="utf-8"))

        self.assertEqual(len(model_payload["fits"]), 2)
        self.assertEqual(prediction_payload["status"], "supported")
        self.assertEqual(prediction_payload["composition"], "serial_upper_bound")
        self.assertAlmostEqual(prediction_payload["predicted_seconds"], 4.74)

    def test_search_space_cli_compiles_audits_and_exports_configuration(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        space = project_root / "examples/conditional-is.deployment-space.example.json"
        capabilities = (
            project_root
            / "examples/npu/vllm-ascend-0.18.capabilities.example.json"
        )
        design = project_root / "examples/conditional-is-short-p96.design.example.json"
        calibration_spec = (
            project_root / "examples/conditional-is-short-p96.calibration.example.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiled_path = root / "compiled.json"
            configuration_path = root / "configuration.json"
            candidate_plan_path = root / "candidate-plan.json"
            configurations_path = root / "candidate-configurations.json"
            updated_spec_path = root / "updated-calibration-spec.json"
            calibration_plan_path = root / "calibration-plan.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "compile-space",
                            str(space),
                            "--capabilities",
                            str(capabilities),
                            "--output",
                            str(compiled_path),
                        ]
                    ),
                    0,
                )
                self.assertEqual(main(["audit-space", str(compiled_path)]), 0)
                self.assertEqual(
                    main(
                        [
                            "plan-candidates",
                            str(design),
                            str(space),
                            str(compiled_path),
                            "--output",
                            str(candidate_plan_path),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(["audit-candidate-plan", str(candidate_plan_path)]),
                    0,
                )
            compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
            candidate_id = compiled["candidates"][0]["candidate_id"]
            self.assertEqual(
                main(
                    [
                        "space-config",
                        str(compiled_path),
                        candidate_id,
                        "--output",
                        str(configuration_path),
                    ]
                ),
                0,
            )
            configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
            self.assertEqual(
                main(
                    [
                        "candidate-configs",
                        str(candidate_plan_path),
                        "--output",
                        str(configurations_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "candidate-calibration-spec",
                        str(calibration_spec),
                        str(candidate_plan_path),
                        "--output",
                        str(updated_spec_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "plan-calibration",
                        str(updated_spec_path),
                        "--output",
                        str(calibration_plan_path),
                    ]
                ),
                0,
            )
            candidate_plan = json.loads(candidate_plan_path.read_text(encoding="utf-8"))
            configurations = json.loads(configurations_path.read_text(encoding="utf-8"))
            updated_spec = json.loads(updated_spec_path.read_text(encoding="utf-8"))
            calibration_plan = json.loads(calibration_plan_path.read_text(encoding="utf-8"))

        self.assertEqual(compiled["audit"]["candidate_count"], 2400)
        self.assertEqual(configuration["configuration_id"], candidate_id)
        self.assertEqual(len(configuration["settings"]), 10)
        self.assertEqual(candidate_plan["audit"]["selected_candidate_count"], 8)
        self.assertEqual(len(configurations["configurations"]), 7)
        self.assertEqual(len(updated_spec["candidates"]), 7)
        self.assertEqual(len(calibration_plan["runs"]), 28)

    def test_graph_capture_cli_plans_and_audits_profile(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        profile = (
            project_root
            / "examples"
            / "conditional-is-base.acl-graph-profile.example.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture-plan.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "plan-graph-capture",
                            str(profile),
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
                self.assertEqual(main(["audit-graph-capture", str(output)]), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["selected_capture_sizes"], [1, 8, 16, 32])
        self.assertGreater(payload["metrics"]["graph_hit_rate"], 0.9)


if __name__ == "__main__":
    unittest.main()
