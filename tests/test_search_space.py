from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from inference_autopilot.adapters import build_conditional_is_small_proposal_graph
from inference_autopilot.search_space import (
    CompiledSearchSpace,
    DeploymentSearchSpace,
    DomainSpec,
    RuntimeCapabilityProfile,
    compile_search_space,
    configuration_from_candidate,
)


def _minimal_space() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "space_id": "minimal-space",
        "algorithm_id": "conditional_is_small_proposal",
        "fixed_settings": {"base_batch_wait_seconds": 0.01},
        "knobs": [
            {
                "name": "base.max_num_seqs",
                "setting_name": "base_max_num_seqs",
                "value_type": "integer",
                "domain": {"kind": "choices", "values": [128, 256]},
                "change_scope": "engine_restart",
                "semantic_effect": "preserves_algorithm",
                "description": "Base capacity",
            }
        ],
        "constraints": [],
    }


class SearchSpaceTest(unittest.TestCase):
    def test_stage_wavefront_space_compiles_only_interpretable_policies(self) -> None:
        root = Path(__file__).resolve().parents[1]
        raw = json.loads(
            (
                root
                / "examples/npu/conditional-is-short-p96-stage-wavefront.deployment-space.json"
            ).read_text(encoding="utf-8")
        )
        graph = build_conditional_is_small_proposal_graph(
            candidate_count=8,
            rollout_count=3,
            block_size=48,
            total_length=192,
        )
        space = DeploymentSearchSpace.from_dict(raw)
        space.validate_against_graph(graph)
        compiled = compile_search_space(space)

        self.assertEqual(compiled.cartesian_product_size, 8)
        self.assertEqual(len(compiled.candidates), 5)
        self.assertEqual(compiled.rejected_candidate_count, 3)
        self.assertEqual(
            {
                candidate.deployment_settings["proposal_stage_wavefront_mode"]
                for candidate in compiled.candidates
            },
            {"off", "auto"},
        )
        self.assertEqual(
            compiled.audit()["by_tuning_layer"], {"hot_policy": 5}
        )

    def test_conditional_is_example_compiles_expected_constrained_grid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        raw = json.loads(
            (root / "examples/conditional-is.deployment-space.example.json").read_text(
                encoding="utf-8"
            )
        )
        capability_raw = json.loads(
            (
                root
                / "examples/npu/vllm-ascend-0.18.capabilities.example.json"
            ).read_text(encoding="utf-8")
        )
        compiled = compile_search_space(
            DeploymentSearchSpace.from_dict(raw),
            capability_profile=RuntimeCapabilityProfile.from_dict(capability_raw),
        )

        self.assertEqual(compiled.cartesian_product_size, 19200)
        self.assertEqual(len(compiled.candidates), 2400)
        self.assertEqual(compiled.rejected_candidate_count, 16800)
        self.assertEqual(
            compiled.rejections_by_constraint,
            {
                "allowed_combinations:base.max_num_seqs,proposal.max_num_seqs": 11520,
                "capability:vllm_ascend.model_runner.mrv2:unsupported": 9600,
                "sum_less_equal:base.memory_fraction,proposal.memory_fraction": 7200,
            },
        )
        self.assertEqual(compiled.audit()["semantic_cohort_count"], 1)
        self.assertEqual(
            compiled.audit()["by_tuning_layer"],
            {"hot_policy": 2400, "static_deployment": 2400},
        )
        self.assertEqual(compiled.audit()["requires_engine_restart_count"], 2400)
        self.assertEqual(
            {candidate.knob_values["runtime.model_runner"] for candidate in compiled.candidates},
            {"MRV1"},
        )
        candidate = compiled.candidates[0]
        self.assertEqual(
            set(candidate.deployment_settings),
            {
                "base_batch_wait_seconds",
                "proposal_batch_wait_seconds",
                "model_runner",
                "base_max_num_seqs",
                "proposal_max_num_seqs",
                "base_max_num_batched_tokens",
                "proposal_max_num_batched_tokens",
                "base_memory_fraction",
                "proposal_memory_fraction",
                "base_score_priority",
            },
        )

    def test_compiled_space_round_trip_rejects_tampering(self) -> None:
        compiled = compile_search_space(
            DeploymentSearchSpace.from_dict(_minimal_space())
        )
        payload = compiled.to_dict()
        self.assertEqual(CompiledSearchSpace.from_dict(payload), compiled)

        payload["candidates"][0]["deployment_settings"]["base_max_num_seqs"] = 1
        with self.assertRaisesRegex(ValueError, "SHA256"):
            CompiledSearchSpace.from_dict(payload)

    def test_capability_requirement_is_fail_closed(self) -> None:
        raw = _minimal_space()
        raw["knobs"][0]["capability_requirements"] = [
            {"capability_id": "runtime.large_batch", "values": [256]}
        ]
        space = DeploymentSearchSpace.from_dict(raw)
        with self.assertRaisesRegex(ValueError, "profile required"):
            compile_search_space(space)

        profile = RuntimeCapabilityProfile.from_dict(
            {
                "schema_version": "1.0",
                "profile_id": "test-capabilities",
                "environment_id": "test-environment",
                "capabilities": {
                    "runtime.large_batch": {
                        "status": "unsupported",
                        "reason": "probe failed",
                        "evidence_sha256": [],
                    }
                },
            }
        )
        compiled = compile_search_space(space, capability_profile=profile)
        self.assertEqual(
            [candidate.knob_values["base.max_num_seqs"] for candidate in compiled.candidates],
            [128],
        )
        self.assertEqual(compiled.capability_profile_sha256, profile.sha256)

    def test_number_range_uses_decimal_steps(self) -> None:
        domain = DomainSpec.from_dict(
            {"kind": "range", "minimum": 0.1, "maximum": 0.3, "step": 0.1}
        )
        self.assertEqual(domain.expand("number"), (0.1, 0.2, 0.3))

    def test_algorithm_changing_knob_requires_opt_in_and_splits_cohorts(self) -> None:
        raw = _minimal_space()
        raw["knobs"].append(
            {
                "name": "algorithm.rollout_count",
                "setting_name": "rollout_count",
                "value_type": "integer",
                "domain": {"kind": "choices", "values": [2, 3]},
                "change_scope": "per_request",
                "semantic_effect": "changes_algorithm",
                "description": "Quality-sensitive rollout budget",
            }
        )
        space = DeploymentSearchSpace.from_dict(raw)

        with self.assertRaisesRegex(ValueError, "explicit opt-in"):
            compile_search_space(space)
        compiled = compile_search_space(space, allow_algorithm_changes=True)

        self.assertEqual(len(compiled.candidates), 4)
        self.assertEqual(compiled.audit()["semantic_cohort_count"], 2)
        self.assertEqual(
            {
                candidate.semantic_settings["rollout_count"]
                for candidate in compiled.candidates
            },
            {2, 3},
        )

    def test_deployment_candidate_exports_to_calibration_configuration(self) -> None:
        compiled = compile_search_space(
            DeploymentSearchSpace.from_dict(_minimal_space())
        )
        candidate = compiled.candidates[0]
        configuration = configuration_from_candidate(compiled, candidate.candidate_id)

        self.assertEqual(configuration.configuration_id, candidate.candidate_id)
        self.assertEqual(configuration.settings, candidate.deployment_settings)

    def test_integer_sequence_knob_compiles_runner_ready_graph_policies(self) -> None:
        raw = _minimal_space()
        raw["fixed_settings"] = {
            "base_batch_wait_seconds": 0.01,
            "base_graph_capture_sizes": [1, 2, 4, 8, 16, 32, 40],
        }
        raw["knobs"].append(
            {
                "name": "proposal.graph_capture_sizes",
                "setting_name": "proposal_graph_capture_sizes",
                "value_type": "integer_sequence",
                "domain": {
                    "kind": "choices",
                    "values": [
                        [1, 2, 4, 8, 16, 24, 32, 40, 48],
                        [1, 2, 4, 8, 16, 24, 32, 40, 48, 64],
                    ],
                },
                "change_scope": "engine_restart",
                "semantic_effect": "preserves_algorithm",
                "tuning_layer": "graph_capture",
                "applies_to_stages": [],
                "description": "Proposal ACL Graph capture buckets",
            }
        )
        compiled = compile_search_space(DeploymentSearchSpace.from_dict(raw))

        self.assertEqual(len(compiled.candidates), 4)
        self.assertEqual(
            {
                candidate.knob_values["proposal.graph_capture_sizes"][-1]
                for candidate in compiled.candidates
            },
            {48, 64},
        )
        payload = compiled.to_dict()
        self.assertIsInstance(
            payload["candidates"][0]["deployment_settings"][
                "proposal_graph_capture_sizes"
            ],
            list,
        )
        self.assertEqual(CompiledSearchSpace.from_dict(payload), compiled)
        configuration = configuration_from_candidate(
            compiled, compiled.candidates[0].candidate_id
        )
        self.assertIsInstance(
            configuration.settings["proposal_graph_capture_sizes"], list
        )

    def test_integer_sequence_rejects_unsorted_or_duplicate_buckets(self) -> None:
        raw = _minimal_space()
        raw["knobs"][0] = {
            **raw["knobs"][0],
            "name": "proposal.graph_capture_sizes",
            "setting_name": "proposal_graph_capture_sizes",
            "value_type": "integer_sequence",
            "domain": {"kind": "choices", "values": [[1, 4, 2], [1, 2, 2]]},
        }
        with self.assertRaisesRegex(ValueError, "integer sequences"):
            DeploymentSearchSpace.from_dict(raw)

    def test_equivalent_domain_order_has_identical_candidate_ids(self) -> None:
        first_raw = _minimal_space()
        second_raw = deepcopy(first_raw)
        second_raw["knobs"][0]["domain"]["values"] = [256, 128]
        first = compile_search_space(DeploymentSearchSpace.from_dict(first_raw))
        second = compile_search_space(DeploymentSearchSpace.from_dict(second_raw))

        self.assertEqual(
            [candidate.candidate_id for candidate in first.candidates],
            [candidate.candidate_id for candidate in second.candidates],
        )

    def test_cartesian_product_limit_fails_before_candidate_generation(self) -> None:
        space = DeploymentSearchSpace.from_dict(_minimal_space())
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            compile_search_space(space, max_cartesian_product=1)

    def test_allowed_combination_outside_domain_is_rejected(self) -> None:
        raw = _minimal_space()
        raw["knobs"].append(
            {
                "name": "proposal.max_num_seqs",
                "setting_name": "proposal_max_num_seqs",
                "value_type": "integer",
                "domain": {"kind": "choices", "values": [768]},
                "change_scope": "engine_restart",
                "semantic_effect": "preserves_algorithm",
                "description": "Proposal capacity",
            }
        )
        raw["constraints"] = [
            {
                "kind": "allowed_combinations",
                "parameters": ["base.max_num_seqs", "proposal.max_num_seqs"],
                "values": [[999, 768]],
            }
        ]
        with self.assertRaisesRegex(ValueError, "outside domain"):
            DeploymentSearchSpace.from_dict(raw)

    def test_tuning_layer_must_match_change_scope(self) -> None:
        raw = _minimal_space()
        raw["knobs"][0]["tuning_layer"] = "hot_policy"
        with self.assertRaisesRegex(ValueError, "cannot use change scope"):
            DeploymentSearchSpace.from_dict(raw)

    def test_stage_bindings_are_checked_against_algorithm_graph(self) -> None:
        root = Path(__file__).resolve().parents[1]
        raw = json.loads(
            (root / "examples/conditional-is.deployment-space.example.json").read_text(
                encoding="utf-8"
            )
        )
        graph = build_conditional_is_small_proposal_graph()
        space = DeploymentSearchSpace.from_dict(raw)
        space.validate_against_graph(graph)

        raw["knobs"][0]["applies_to_stages"] = ["candidate_select"]
        with self.assertRaisesRegex(ValueError, "not declared tunable"):
            DeploymentSearchSpace.from_dict(raw).validate_against_graph(graph)


if __name__ == "__main__":
    unittest.main()
