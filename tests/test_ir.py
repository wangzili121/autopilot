from __future__ import annotations

import unittest

from inference_autopilot.adapters import build_conditional_is_small_proposal_graph
from inference_autopilot.ir import InferenceGraph, IntExpr, LoopSpec, ParameterSpec, StageSpec


class IntExprTest(unittest.TestCase):
    def test_structured_expression_evaluates(self) -> None:
        expression = IntExpr.compound(
            "multiply", IntExpr.symbol("candidates"), IntExpr.symbol("rollouts")
        )
        self.assertEqual(expression.evaluate({"candidates": 4, "rollouts": 3}), 12)
        self.assertEqual(expression.symbols(), frozenset(("candidates", "rollouts")))


class ConditionalISGraphTest(unittest.TestCase):
    def test_exact_graph_preserves_small_proposal_dependencies(self) -> None:
        graph = build_conditional_is_small_proposal_graph(
            candidate_count=4,
            rollout_count=3,
            block_size=16,
            total_length=128,
        )
        stages = {stage.stage_id: stage for stage in graph.stages}
        self.assertEqual(graph.algorithm_semantics, "exact")
        self.assertEqual(
            stages["proposal_rollout_generate"].multiplicity_upper_bound.evaluate(graph.bindings),
            12,
        )
        self.assertEqual(
            set(stages["importance_reduce"].depends_on),
            {"target_score", "reward_evaluate"},
        )
        self.assertEqual(graph.loop.maximum_iterations.evaluate(graph.bindings), 8)

    def test_uncorrected_graph_is_explicitly_biased(self) -> None:
        graph = build_conditional_is_small_proposal_graph(apply_importance_correction=False)
        self.assertEqual(graph.algorithm_semantics, "biased_ablation")
        self.assertNotIn("target_score", {stage.stage_id for stage in graph.stages})

    def test_graph_round_trip_preserves_structured_expressions(self) -> None:
        graph = build_conditional_is_small_proposal_graph(
            candidate_count=5, rollout_count=2
        )
        restored = InferenceGraph.from_dict(graph.to_dict())

        self.assertEqual(restored, graph)
        self.assertEqual(
            restored.stages[1].multiplicity_upper_bound.evaluate(restored.bindings),
            10,
        )

    def test_graph_rejects_block_larger_than_total_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "block_size"):
            build_conditional_is_small_proposal_graph(block_size=129, total_length=128)

    def test_graph_rejects_unknown_dependency(self) -> None:
        parameter = ParameterSpec(
            "count", "integer", 1, "algorithm_budget", "test", minimum=1
        )
        stage = StageSpec(
            "only_stage",
            "generate",
            "base_model",
            IntExpr.symbol("count"),
            IntExpr.literal(1),
            depends_on=("missing",),
        )
        with self.assertRaisesRegex(ValueError, "unknown dependencies"):
            InferenceGraph(
                algorithm_id="test",
                algorithm_semantics="exact",
                parameters=(parameter,),
                stages=(stage,),
                loop=LoopSpec(("only_stage",), IntExpr.literal(1), "count", "increment"),
                source_contract=("test",),
            )


if __name__ == "__main__":
    unittest.main()
