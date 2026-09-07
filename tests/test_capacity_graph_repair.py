from __future__ import annotations

from copy import deepcopy
import unittest

from inference_autopilot.calibration.models import CalibrationSpec
from inference_autopilot.capacity_graph_repair import (
    CapacityGraphRepairPlan,
    calibration_spec_from_capacity_graph_repair,
    plan_capacity_graph_repair,
)


def _calibration_spec() -> CalibrationSpec:
    common = {
        "base_max_num_seqs": 40,
        "base_graph_mode": "FULL_DECODE_ONLY",
        "base_graph_capture_sizes": [1, 40],
        "proposal_graph_mode": "FULL_DECODE_ONLY",
        "proposal_graph_capture_sizes": [1, 48],
    }
    return CalibrationSpec.from_dict(
        {
            "schema_version": "1.0",
            "campaign_id": "capacity-probe",
            "strong_baseline": True,
            "protocol": {"pattern": "ABBA", "blocks": 1, "pair_seeds": [1, 2]},
            "semantic_contract": {
                "algorithm_id": "conditional_is_small_proposal",
                "semantic_class": "exact",
                "graph_sha256": "a" * 64,
                "invariants": {"candidate_count": 8},
            },
            "workload_contract": {
                "workload_id": "short-p16",
                "dataset_sha256": "b" * 64,
                "arrival_trace_sha256": "c" * 64,
                "parameters": {"requests": 16},
            },
            "environment_contract": {
                "environment_id": "npu-test",
                "hardware": {"device": "Ascend"},
                "software": {"vllm": "0.18"},
                "models": {"proposal": "test"},
            },
            "objective": {
                "primary_metric": "completed_qps",
                "direction": "maximize",
                "constraints": [],
            },
            "required_metrics": ["completed_qps"],
            "baseline": {
                "configuration_id": "active-40-48",
                "settings": {**common, "proposal_max_num_seqs": 48},
                "description": "active fallback",
            },
            "candidates": [
                {
                    "configuration_id": "capacity-64-graph-48",
                    "settings": {**common, "proposal_max_num_seqs": 64},
                    "description": "capacity probe",
                }
            ],
        }
    )


def _assessment(spec: CalibrationSpec, observed_maximum: float = 64.0) -> dict:
    candidate = spec.candidates[0]
    group_id = "capacity-probe--capacity-64-graph-48"
    pair_effects = [
        {
            "pair_index": index,
            "workload_seed": seed,
            "baseline_run_id": f"baseline-{index}",
            "candidate_run_id": f"candidate-{index}",
            "baseline_value": baseline,
            "candidate_value": value,
            "candidate_over_baseline_ratio": value / baseline,
            "directional_relative_improvement": value / baseline - 1,
        }
        for index, (seed, baseline, value) in enumerate(
            ((1, 0.28, 0.20), (2, 0.26, 0.20))
        )
    ]
    records = [
        {
            "record_id": f"candidate-record-{index}",
            "source": {"locator": f"candidate-{index}"},
            "variant": candidate.configuration_id,
            "tags": ["calibration", "candidate", "formal_group"],
            "workload": {"comparison_group_id": group_id},
            "configuration": {
                "configuration_id": candidate.configuration_id,
                **dict(candidate.settings),
            },
            "metrics": {
                "vllm_runtime_metrics": {
                    "proposal": {
                        "vllm:num_requests_running": {
                            "maximum": observed_maximum,
                        }
                    }
                }
            },
        }
        for index in range(2)
    ]
    return {
        "schema_version": "1.0",
        "plan_sha256": "d" * 64,
        "formal_complete": True,
        "effects": [
            {
                "comparison_group_id": group_id,
                "candidate_configuration_id": candidate.configuration_id,
                "primary_metric": "completed_qps",
                "direction": "maximize",
                "replay_control": False,
                "formal_group": True,
                "quality_constraints_satisfied": True,
                "expected_pair_count": 2,
                "complete_pair_count": 2,
                "pair_effects": pair_effects,
                "median_directional_relative_improvement": -0.25,
                "effect_outside_replay_noise": True,
            }
        ],
        "replay_noise_envelope": 0.04,
        "ledger": {"records": records},
    }


class CapacityGraphRepairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = _calibration_spec()
        self.assessment = _assessment(self.spec)

    def test_plans_graph_domain_repair_from_formal_regression(self) -> None:
        plan = plan_capacity_graph_repair(
            self.assessment,
            self.spec,
            repair_id="proposal-graph-repair-64",
        )
        payload = plan.to_dict()

        self.assertEqual(plan.audit()["engine_roles"], ["proposal"])
        self.assertEqual(payload["diagnoses"][0]["capture_ceiling"], 48)
        self.assertEqual(payload["diagnoses"][0]["observed_running_max"], 64.0)
        self.assertEqual(payload["diagnoses"][0]["target_capture_size"], 64)
        self.assertEqual(
            payload["repair"]["repaired_candidate_configuration"]["settings"][
                "proposal_graph_capture_sizes"
            ],
            [1, 48, 64],
        )
        self.assertEqual(
            [item["setting"] for item in payload["repair"]["repair_delta"]],
            ["proposal_graph_capture_sizes"],
        )
        self.assertEqual(
            [item["setting"] for item in payload["repair"]["production_delta"]],
            ["proposal_graph_capture_sizes", "proposal_max_num_seqs"],
        )
        self.assertEqual(CapacityGraphRepairPlan.from_dict(payload).to_dict(), payload)

    def test_compiles_repair_against_active_baseline_with_replay(self) -> None:
        plan = plan_capacity_graph_repair(
            self.assessment,
            self.spec,
            repair_id="proposal-graph-repair-64",
        )
        calibration = calibration_spec_from_capacity_graph_repair(
            self.spec,
            plan,
            campaign_id="proposal-graph-repair-64-r1",
            pair_seeds=(11, 22),
        )

        self.assertEqual(calibration.protocol.pair_seeds, (11, 22))
        self.assertEqual(calibration.baseline, self.spec.baseline)
        self.assertEqual(len(calibration.candidates), 2)
        self.assertEqual(calibration.candidates[0].settings, calibration.baseline.settings)
        self.assertEqual(calibration.candidates[1].settings["proposal_max_num_seqs"], 64)
        self.assertEqual(
            calibration.candidates[1].settings["proposal_graph_capture_sizes"],
            [1, 48, 64],
        )

    def test_rejects_regression_without_observed_graph_domain_breach(self) -> None:
        assessment = _assessment(self.spec, observed_maximum=48.0)
        with self.assertRaisesRegex(ValueError, "capacity/graph-domain breach"):
            plan_capacity_graph_repair(
                assessment,
                self.spec,
                repair_id="proposal-graph-repair-64",
            )

    def test_rejects_effect_inside_replay_noise(self) -> None:
        assessment = deepcopy(self.assessment)
        assessment["effects"][0]["effect_outside_replay_noise"] = False
        with self.assertRaisesRegex(ValueError, "outside replay noise"):
            plan_capacity_graph_repair(
                assessment,
                self.spec,
                repair_id="proposal-graph-repair-64",
            )

    def test_rejects_plan_tampering(self) -> None:
        payload = plan_capacity_graph_repair(
            self.assessment,
            self.spec,
            repair_id="proposal-graph-repair-64",
        ).to_dict()
        payload["diagnoses"][0]["target_capture_size"] = 96
        with self.assertRaisesRegex(ValueError, "SHA256"):
            CapacityGraphRepairPlan.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
