from __future__ import annotations

import unittest

from inference_autopilot.graph_bucket_search import (
    GraphBucketConstraints,
    build_coverage_constrained_graph_experiment_plan,
    search_coverage_constrained_graph_policy,
)
from inference_autopilot.graph_corpus import build_graph_profile_corpus
from inference_autopilot.vllm_graph_metrics import parse_vllm_graph_metrics


_LOG = """
[inference-autopilot] graph-stats-begin role=base
INFO x x [graph.py:1] - Mode: FULL_DECODE_ONLY
INFO x x [graph.py:1] - Capture sizes: [1, 2, 4, 8]
INFO x x [graph.py:1] | 1 | 1 | 0 | FULL | 2 |
INFO x x [graph.py:1] | 7 | 8 | 1 | FULL | 18 |
[inference-autopilot] graph-stats-end role=base
[inference-autopilot] graph-stats-begin role=proposal
INFO x x [graph.py:1] - Mode: FULL_DECODE_ONLY
INFO x x [graph.py:1] - Capture sizes: [1, 4, 8, 16]
INFO x x [graph.py:1] | 4 | 4 | 0 | FULL | 4 |
INFO x x [graph.py:1] | 15 | 16 | 1 | FULL | 16 |
[inference-autopilot] graph-stats-end role=proposal
"""


def _profile(profile_id: str, high_count: int):
    log = _LOG.replace("| 7 | 8 | 1 | FULL | 18 |", f"| 7 | 8 | 1 | FULL | {high_count} |")
    return parse_vllm_graph_metrics(log, profile_id=profile_id)


def _corpus(holdout=None):
    return build_graph_profile_corpus(
        corpus_id="bucket-corpus",
        train_profiles=(
            ("load8", _profile("train8", 18)),
            ("load32", _profile("train32", 19)),
        ),
        holdout_profiles=(("load96", holdout or _profile("holdout96", 20)),),
    )


def _large_profile(profile_id: str, event_count: int):
    sizes = list(range(1, 52))
    rows = "\n".join(
        f"INFO x x [graph.py:1] | {size} | {size} | 0 | FULL | {event_count} |"
        for size in sizes
    )
    section = (
        "[inference-autopilot] graph-stats-begin role={role}\n"
        "INFO x x [graph.py:1] - Mode: FULL_DECODE_ONLY\n"
        f"INFO x x [graph.py:1] - Capture sizes: {sizes}\n"
        f"{rows}\n"
        "[inference-autopilot] graph-stats-end role={role}\n"
    )
    return parse_vllm_graph_metrics(
        section.format(role="base") + section.format(role="proposal"),
        profile_id=profile_id,
    )


class GraphBucketSearchTest(unittest.TestCase):
    def test_search_minimizes_bucket_count_and_builds_corpus_bound_plan(self) -> None:
        corpus = _corpus()
        search = search_coverage_constrained_graph_policy(
            corpus,
            policy_id="bounded-remap",
            constraints=GraphBucketConstraints(
                minimum_retained_graph_fraction=1.0,
                maximum_remapped_graph_fraction=0.25,
                maximum_added_padding_ratio=0.1,
            ),
            minimum_train_regimes=2,
            minimum_holdout_regimes=1,
        )
        plan = build_coverage_constrained_graph_experiment_plan(
            corpus,
            search,
            campaign_id="bounded-remap-abba",
            blocks=2,
            pair_seeds=(11, 22, 33, 44),
        )

        self.assertTrue(search.eligible)
        self.assertEqual(search.rejection_reasons, ())
        self.assertEqual(
            {
                item.engine_role: item.search_space_subset_count
                for item in search.role_searches
            },
            {"base": 15, "proposal": 15},
        )
        self.assertTrue(
            all(
                item.search_strategy == "ordered_pareto_dynamic_programming"
                and item.evaluated_transition_count > 0
                and item.retained_pareto_state_count > 0
                for item in search.role_searches
            )
        )
        selected = {
            engine.engine_role: engine.capture_sizes for engine in search.policy.engines
        }
        self.assertEqual(selected, {"base": (1, 8), "proposal": (4, 16)})
        self.assertEqual(plan.source_profile_sha256, corpus.digest)
        self.assertEqual(len(plan.runs), 16)
        self.assertEqual(
            {run.policy_id for run in plan.runs[:8]}, {"vllm-default"}
        )

    def test_search_rejects_candidate_that_violates_holdout_remap_budget(self) -> None:
        holdout_log = _LOG.replace(
            "INFO x x [graph.py:1] | 7 | 8 | 1 | FULL | 18 |",
            "INFO x x [graph.py:1] | 3 | 4 | 1 | FULL | 100 |\n"
            "INFO x x [graph.py:1] | 7 | 8 | 1 | FULL | 18 |",
        )
        holdout = parse_vllm_graph_metrics(holdout_log, profile_id="holdout96")
        search = search_coverage_constrained_graph_policy(
            _corpus(holdout),
            policy_id="bounded-remap",
            constraints=GraphBucketConstraints(
                minimum_retained_graph_fraction=1.0,
                maximum_remapped_graph_fraction=0.25,
                maximum_added_padding_ratio=0.1,
            ),
            minimum_train_regimes=2,
            minimum_holdout_regimes=1,
        )

        self.assertFalse(search.eligible)
        self.assertEqual(
            search.rejection_reasons, ("holdout_constraint_violation",)
        )
        holdout_row = next(
            row for row in search.coverage if row["split"] == "holdout"
        )
        self.assertFalse(holdout_row["passes_constraints"])

    def test_search_uses_independent_holdout_constraints(self) -> None:
        holdout_log = _LOG.replace(
            "INFO x x [graph.py:1] | 7 | 8 | 1 | FULL | 18 |",
            "INFO x x [graph.py:1] | 3 | 4 | 1 | FULL | 100 |\n"
            "INFO x x [graph.py:1] | 7 | 8 | 1 | FULL | 18 |",
        )
        holdout = parse_vllm_graph_metrics(holdout_log, profile_id="holdout96")
        holdout_constraints = GraphBucketConstraints(
            minimum_retained_graph_fraction=1.0,
            maximum_remapped_graph_fraction=0.9,
            maximum_added_padding_ratio=1.0,
        )
        search = search_coverage_constrained_graph_policy(
            _corpus(holdout),
            policy_id="bounded-remap",
            constraints=GraphBucketConstraints(
                minimum_retained_graph_fraction=1.0,
                maximum_remapped_graph_fraction=0.25,
                maximum_added_padding_ratio=0.1,
            ),
            holdout_constraints=holdout_constraints,
            minimum_train_regimes=2,
            minimum_holdout_regimes=1,
        )

        self.assertTrue(search.eligible)
        self.assertEqual(search.holdout_constraints, holdout_constraints)
        self.assertEqual(
            search.to_dict()["holdout_constraints"], holdout_constraints.to_dict()
        )
        holdout_row = next(
            row for row in search.coverage if row["split"] == "holdout"
        )
        self.assertTrue(holdout_row["passes_constraints"])

    def test_search_can_scope_a_candidate_to_one_engine_role(self) -> None:
        search = search_coverage_constrained_graph_policy(
            _corpus(),
            policy_id="proposal-only",
            constraints=GraphBucketConstraints(
                minimum_retained_graph_fraction=1.0,
                maximum_remapped_graph_fraction=0.25,
                maximum_added_padding_ratio=0.1,
            ),
            candidate_engine_roles=("proposal",),
            minimum_train_regimes=2,
            minimum_holdout_regimes=1,
        )

        self.assertTrue(search.eligible)
        self.assertEqual(
            [role_search.engine_role for role_search in search.role_searches],
            ["proposal"],
        )
        selected = {
            engine.engine_role: engine.capture_sizes for engine in search.policy.engines
        }
        self.assertEqual(
            selected,
            {"base": (1, 2, 4, 8), "proposal": (4, 16)},
        )

    def test_ordered_dp_handles_a_realistic_bucket_count(self) -> None:
        corpus = build_graph_profile_corpus(
            corpus_id="large-bucket-corpus",
            train_profiles=(
                ("load8", _large_profile("large-train8", 1)),
                ("load32", _large_profile("large-train32", 2)),
            ),
            holdout_profiles=(
                ("load96", _large_profile("large-holdout96", 3)),
            ),
        )
        search = search_coverage_constrained_graph_policy(
            corpus,
            policy_id="zero-remap",
            constraints=GraphBucketConstraints(
                minimum_retained_graph_fraction=1.0,
                maximum_remapped_graph_fraction=0.0,
                maximum_added_padding_ratio=0.0,
            ),
            minimum_train_regimes=2,
            minimum_holdout_regimes=1,
        )

        self.assertFalse(search.eligible)
        self.assertEqual(search.rejection_reasons, ("no_policy_change",))
        for role_search in search.role_searches:
            self.assertEqual(role_search.configured_bucket_count, 51)
            self.assertEqual(role_search.search_space_subset_count, (1 << 51) - 1)
            self.assertLess(
                role_search.evaluated_transition_count,
                role_search.search_space_subset_count,
            )
            self.assertEqual(role_search.selected_capture_sizes, tuple(range(1, 52)))


if __name__ == "__main__":
    unittest.main()
