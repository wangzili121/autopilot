from __future__ import annotations

import unittest

from inference_autopilot.calibration.models import canonical_sha256
from inference_autopilot.sequential_design import (
    SequentialDesignStudy,
    SequentialDesignStudySpec,
    _build_payload,
    hedged_capital_interval,
)
from inference_autopilot.sequential_effect import (
    SequentialEffectSpec,
    confidence_interval,
)


def _effect_spec() -> SequentialEffectSpec:
    return SequentialEffectSpec(
        effect_id="design-test",
        analysis_mode="prospective",
        baseline_configuration_id="baseline",
        replay_control_configuration_id="replay",
        candidate_configuration_id="candidate",
        primary_metric="completed_qps",
        direction="maximize",
        algorithm_context_sha256="a" * 64,
        workload_context_sha256="b" * 64,
        environment_context_sha256="c" * 64,
        baseline_settings_sha256="d" * 64,
        candidate_settings_sha256="e" * 64,
        lower_log_effect_bound=-0.2,
        upper_log_effect_bound=0.2,
        confidence_alpha=0.05,
        confidence_method="betting_mixture",
        betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.7, 0.9),
        minimum_improvement_fraction=0.03,
        minimum_decision_pairs=4,
        maximum_pair_count=8,
        replication_batch_pairs=2,
        replication_seed_pool=tuple(range(20)),
        require_quality_constraints=True,
    )


def _study_spec(trials: int = 20) -> SequentialDesignStudySpec:
    return SequentialDesignStudySpec(
        study_id="design-study-test",
        methods=(
            "frozen_spec_method",
            "finite_horizon_hoeffding",
            "hedged_capital_predictable_plugin",
        ),
        pair_looks=(4, 6, 8),
        true_log_effects=(-0.05, 0.0, 0.05),
        noise_models=("observed_symmetric_residual", "bounded_endpoints"),
        simulation_trials=trials,
        simulation_seed=7,
    )


class SequentialDesignTest(unittest.TestCase):
    def test_hedged_capital_adapts_to_low_observed_variance(self) -> None:
        spec = _effect_spec()
        values = [-0.0595, -0.0473, 0.0376, 0.0238]
        projected = [*values, *([sum(values) / len(values)] * 4)]

        frozen = confidence_interval(projected, spec)
        hedged = hedged_capital_interval(projected, spec)

        self.assertLess(hedged[1] - hedged[0], frozen[1] - frozen[0])
        self.assertLessEqual(hedged[0], sum(projected) / len(projected))
        self.assertGreaterEqual(hedged[1], sum(projected) / len(projected))

    def test_study_is_deterministic_and_auditable(self) -> None:
        values = [-0.0595, -0.0473, 0.0376, 0.0238]
        payload = _build_payload(
            _study_spec(), _effect_spec(), values, "f" * 64, "replicate"
        )
        repeated = _build_payload(
            _study_spec(), _effect_spec(), values, "f" * 64, "replicate"
        )
        study = SequentialDesignStudy.from_dict(payload)

        self.assertEqual(payload, repeated)
        self.assertEqual(study.status, "diagnostic_only")
        self.assertFalse(study.audit()["formal_method_change_allowed"])
        self.assertEqual(len(payload["simulation_results"]), 18)
        for row in payload["simulation_results"]:
            self.assertAlmostEqual(sum(row["decision_rates"].values()), 1.0)

    def test_rehashed_derived_result_tampering_is_rejected(self) -> None:
        payload = _build_payload(
            _study_spec(5),
            _effect_spec(),
            [-0.0595, -0.0473, 0.0376, 0.0238],
            "f" * 64,
            "replicate",
        )
        payload["observed_method_comparison"][0]["projected_boundary"] = "promote"
        unsigned = dict(payload)
        unsigned.pop("sequential_design_study_sha256")
        payload["sequential_design_study_sha256"] = canonical_sha256(unsigned)

        with self.assertRaisesRegex(ValueError, "derived results"):
            SequentialDesignStudy.from_dict(payload)

    def test_symmetric_residual_support_is_enforced(self) -> None:
        spec = _study_spec(1)
        invalid = SequentialDesignStudySpec(
            study_id=spec.study_id,
            methods=spec.methods,
            pair_looks=spec.pair_looks,
            true_log_effects=(0.19,),
            noise_models=("observed_symmetric_residual",),
            simulation_trials=1,
            simulation_seed=spec.simulation_seed,
        )
        with self.assertRaisesRegex(ValueError, "symmetric residual"):
            _build_payload(
                invalid,
                _effect_spec(),
                [-0.0595, -0.0473, 0.0376, 0.0238],
                "f" * 64,
                "replicate",
            )


if __name__ == "__main__":
    unittest.main()
