from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from inference_autopilot.cli import main
from inference_autopilot.calibration.models import CalibrationSpec
from inference_autopilot.graph_experiment import (
    GraphExperimentPlan,
    apply_graph_policy_to_config,
    build_graph_experiment_plan,
    build_graph_policy_calibration_spec,
    evaluate_graph_policy_coverage,
)
from inference_autopilot.vllm_graph_metrics import parse_vllm_graph_metrics


_PROFILE_LOG = """
[inference-autopilot] graph-stats-begin role=base
INFO 09-04 09:32:40 [cuda_graph.py:123] - Mode: FULL_DECODE_ONLY
INFO 09-04 09:32:40 [cuda_graph.py:123] - Capture sizes: [1, 2, 4, 8]
INFO 09-04 09:32:40 [cuda_graph.py:123] | 1 | 1 | 0 | FULL | 8 |
INFO 09-04 09:32:40 [cuda_graph.py:123] | 7 | 8 | 1 | FULL | 5 |
[inference-autopilot] graph-stats-end role=base
[inference-autopilot] graph-stats-begin role=proposal
INFO 09-04 09:32:40 [cuda_graph.py:123] - Mode: FULL_DECODE_ONLY
INFO 09-04 09:32:40 [cuda_graph.py:123] - Capture sizes: [1, 4, 8, 16]
INFO 09-04 09:32:40 [cuda_graph.py:123] | 4 | 4 | 0 | FULL | 4 |
INFO 09-04 09:32:40 [cuda_graph.py:123] | 15 | 16 | 1 | FULL | 9 |
[inference-autopilot] graph-stats-end role=proposal
"""


class GraphExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = parse_vllm_graph_metrics(
            _PROFILE_LOG, profile_id="graph-profile"
        )

    def test_builds_separate_pruned_and_no_graph_abba_groups(self) -> None:
        plan = build_graph_experiment_plan(
            self.profile,
            campaign_id="graph-abba",
            blocks=1,
            pair_seeds=(11, 22),
        )

        self.assertEqual(
            [policy.policy_id for policy in plan.policies],
            ["vllm-default", "trace-preserving-pruned", "no-graph"],
        )
        self.assertEqual(len(plan.runs), 8)
        self.assertEqual(
            [run.policy_id for run in plan.runs[:4]],
            [
                "vllm-default",
                "trace-preserving-pruned",
                "trace-preserving-pruned",
                "vllm-default",
            ],
        )
        self.assertEqual([run.workload_seed for run in plan.runs[:4]], [11, 11, 22, 22])
        pruned = plan.policies[1]
        sizes = {engine.engine_role: engine.capture_sizes for engine in pruned.engines}
        self.assertEqual(sizes, {"base": (1, 8), "proposal": (4, 16)})

    def test_policy_injection_is_role_specific_and_preserves_other_settings(self) -> None:
        plan = build_graph_experiment_plan(
            self.profile,
            campaign_id="graph-abba",
            blocks=1,
            pair_seeds=(11, 22),
        )
        config = {
            "run": {"seed": 1, "subset_seed": 7},
            "vllm": {
                "engine_kwargs": {
                    "enable_chunked_prefill": True,
                    "compilation_config": {"cudagraph_mode": "OLD"},
                },
                "base": {"engine_kwargs": {"scheduling_policy": "priority"}},
                "proposal": {},
            },
        }
        apply_graph_policy_to_config(
            config,
            plan.policies[1],
            workload_seed=22,
        )

        self.assertNotIn("compilation_config", config["vllm"]["engine_kwargs"])
        self.assertTrue(config["vllm"]["engine_kwargs"]["enable_chunked_prefill"])
        self.assertEqual(config["run"], {"seed": 22, "subset_seed": 7})
        self.assertEqual(
            config["vllm"]["base"]["engine_kwargs"],
            {
                "scheduling_policy": "priority",
                "compilation_config": {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [1, 8],
                },
            },
        )

    def test_optional_replay_control_uses_default_policy_on_both_arms(self) -> None:
        plan = build_graph_experiment_plan(
            self.profile,
            campaign_id="graph-abba",
            blocks=1,
            pair_seeds=(11, 22),
            include_replay_control=True,
        )

        self.assertEqual(len(plan.runs), 12)
        self.assertEqual(
            {run.comparison_id for run in plan.runs[:4]},
            {"graph-abba--replay-control"},
        )
        self.assertEqual(
            [run.policy_id for run in plan.runs[:4]], ["vllm-default"] * 4
        )

    def test_builds_manifest_bound_calibration_from_graph_policies(self) -> None:
        graph_plan = build_graph_experiment_plan(
            self.profile,
            campaign_id="graph-abba",
            blocks=1,
            pair_seeds=(11, 22),
        )
        template = CalibrationSpec.from_dict(
            json.loads(
                Path(
                    "examples/conditional-is-short-p96.calibration.example.json"
                ).read_text(encoding="utf-8")
            )
        )
        spec = build_graph_policy_calibration_spec(
            template,
            graph_plan,
            campaign_id="formal-graph-abba",
            candidate_policy_ids=("trace-preserving-pruned",),
            include_replay_control=True,
        )

        self.assertEqual(spec.campaign_id, "formal-graph-abba")
        self.assertEqual(spec.workload_contract, template.workload_contract)
        self.assertEqual(spec.environment_contract, template.environment_contract)
        self.assertEqual(
            spec.baseline.settings["proposal_graph_capture_sizes"],
            [1, 4, 8, 16],
        )
        self.assertEqual(
            [candidate.configuration_id for candidate in spec.candidates],
            [
                "manual-256-896-graph-replay-control",
                "manual-256-896-graph-trace-preserving-pruned",
            ],
        )
        self.assertEqual(spec.candidates[0].settings, spec.baseline.settings)
        self.assertEqual(
            spec.candidates[1].settings["proposal_graph_capture_sizes"],
            [4, 16],
        )

    def test_holdout_coverage_detects_bucket_remapping(self) -> None:
        plan = build_graph_experiment_plan(
            self.profile,
            campaign_id="graph-abba",
            blocks=1,
            pair_seeds=(11, 22),
        )
        holdout_log = _PROFILE_LOG.replace(
            "[inference-autopilot] graph-stats-end role=base",
            "INFO 09-04 09:32:40 [cuda_graph.py:123] | 3 | 4 | 1 | FULL | 6 |\n"
            "[inference-autopilot] graph-stats-end role=base",
        )
        holdout = parse_vllm_graph_metrics(holdout_log, profile_id="holdout")
        default = evaluate_graph_policy_coverage(plan.policies[0], holdout)
        pruned = evaluate_graph_policy_coverage(plan.policies[1], holdout)
        base = next(
            engine for engine in pruned.engines if engine["engine_role"] == "base"
        )

        self.assertTrue(default.mapping_exact)
        self.assertFalse(pruned.mapping_exact)
        self.assertEqual(base["remapped_graph_event_count"], 6)
        self.assertEqual(base["dropped_graph_event_count"], 0)
        self.assertEqual(base["additional_padding_units"], 24)
        self.assertIn("graph_policy_coverage_sha256", pruned.artifact_dict())

    def test_plan_round_trip_rejects_tampering(self) -> None:
        artifact = build_graph_experiment_plan(
            self.profile,
            campaign_id="graph-abba",
            blocks=1,
            pair_seeds=(11, 22),
        ).artifact_dict()
        self.assertEqual(GraphExperimentPlan.from_dict(artifact).artifact_dict(), artifact)
        tampered = deepcopy(artifact)
        tampered["runs"][0]["workload_seed"] = 999
        with self.assertRaisesRegex(ValueError, "SHA256"):
            GraphExperimentPlan.from_dict(tampered)

    def test_requires_exact_seed_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "two pair seeds"):
            build_graph_experiment_plan(
                self.profile,
                campaign_id="graph-abba",
                blocks=2,
                pair_seeds=(11, 22),
            )

    def test_cli_plans_and_audits_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            plan_path = root / "plan.json"
            profile_path.write_text(
                json.dumps(self.profile.artifact_dict()), encoding="utf-8"
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "plan-vllm-graph-experiment",
                            str(profile_path),
                            "--campaign-id",
                            "graph-abba",
                            "--blocks",
                            "1",
                            "--pair-seeds",
                            "11",
                            "22",
                            "--output",
                            str(plan_path),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(["audit-vllm-graph-experiment", str(plan_path)]), 0
                )
            artifact = json.loads(plan_path.read_text(encoding="utf-8"))

        self.assertEqual(len(artifact["runs"]), 8)
        self.assertIn("graph_experiment_plan_sha256", artifact)


if __name__ == "__main__":
    unittest.main()
