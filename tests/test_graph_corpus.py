from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from inference_autopilot.cli import main
from inference_autopilot.graph_corpus import (
    GraphProfileCorpus,
    build_graph_profile_corpus,
    build_promoted_graph_experiment_plan,
    promote_trace_preserving_graph_policy,
)
from inference_autopilot.vllm_graph_metrics import parse_vllm_graph_metrics


_LOG = """
[inference-autopilot] graph-stats-begin role=base
INFO x x [graph.py:1] - Mode: FULL_DECODE_ONLY
INFO x x [graph.py:1] - Capture sizes: [1, 2, 4, 8]
INFO x x [graph.py:1] | 1 | 1 | 0 | FULL | 8 |
INFO x x [graph.py:1] | 7 | 8 | 1 | FULL | 5 |
[inference-autopilot] graph-stats-end role=base
[inference-autopilot] graph-stats-begin role=proposal
INFO x x [graph.py:1] - Mode: FULL_DECODE_ONLY
INFO x x [graph.py:1] - Capture sizes: [1, 4, 8, 16]
INFO x x [graph.py:1] | 4 | 4 | 0 | FULL | 4 |
INFO x x [graph.py:1] | 15 | 16 | 1 | FULL | 9 |
[inference-autopilot] graph-stats-end role=proposal
"""


def _profile(profile_id: str, count: int):
    log = _LOG.replace("| 7 | 8 | 1 | FULL | 5 |", f"| 7 | 8 | 1 | FULL | {count} |")
    return parse_vllm_graph_metrics(log, profile_id=profile_id)


class GraphCorpusTest(unittest.TestCase):
    def test_corpus_is_content_addressed_and_rejects_regime_leakage(self) -> None:
        corpus = build_graph_profile_corpus(
            corpus_id="load-corpus",
            train_profiles=(("load8", _profile("train", 5)),),
            holdout_profiles=(("load32", _profile("holdout", 6)),),
        )
        artifact = corpus.artifact_dict()

        self.assertEqual(GraphProfileCorpus.from_dict(artifact).artifact_dict(), artifact)
        self.assertEqual(corpus.audit()["splits"]["train"]["regime_ids"], ["load8"])
        with self.assertRaisesRegex(ValueError, "leaks across splits"):
            build_graph_profile_corpus(
                corpus_id="leaky",
                train_profiles=(("same-load", _profile("train", 5)),),
                holdout_profiles=(("same-load", _profile("holdout", 6)),),
            )

    def test_promotion_requires_independent_regimes_and_exact_holdout_mapping(self) -> None:
        holdout_log = _LOG.replace(
            "[inference-autopilot] graph-stats-end role=base",
            "INFO x x [graph.py:1] | 3 | 4 | 1 | FULL | 6 |\n"
            "[inference-autopilot] graph-stats-end role=base",
        )
        corpus = build_graph_profile_corpus(
            corpus_id="load-corpus",
            train_profiles=(
                ("load8", _profile("train8", 5)),
                ("load32", _profile("train32", 6)),
            ),
            holdout_profiles=(
                (
                    "load96",
                    parse_vllm_graph_metrics(holdout_log, profile_id="holdout96"),
                ),
            ),
        )

        promotion = promote_trace_preserving_graph_policy(
            corpus,
            minimum_train_regimes=2,
            minimum_holdout_regimes=1,
        )

        self.assertFalse(promotion.eligible)
        self.assertEqual(promotion.rejection_reasons, ("holdout_mapping_not_exact",))
        holdout = next(row for row in promotion.coverage if row["split"] == "holdout")
        self.assertFalse(holdout["coverage"]["mapping_exact"])

    def test_eligible_policy_builds_corpus_bound_abba_with_replay_control(self) -> None:
        corpus = build_graph_profile_corpus(
            corpus_id="load-corpus",
            train_profiles=(
                ("load8", _profile("train8", 5)),
                ("load32", _profile("train32", 6)),
            ),
            holdout_profiles=(("load96", _profile("holdout96", 7)),),
        )
        promotion = promote_trace_preserving_graph_policy(
            corpus,
            minimum_train_regimes=2,
            minimum_holdout_regimes=1,
        )
        plan = build_promoted_graph_experiment_plan(
            corpus,
            promotion,
            campaign_id="robust-abba",
            blocks=2,
            pair_seeds=(11, 22, 33, 44),
        )

        self.assertTrue(promotion.eligible)
        self.assertEqual(promotion.rejection_reasons, ())
        self.assertEqual(plan.source_profile_sha256, corpus.digest)
        self.assertEqual(len(plan.runs), 24)
        self.assertEqual(
            {run.policy_id for run in plan.runs[:8]},
            {"vllm-default"},
        )

    def test_cli_builds_corpus_and_rejects_a_noop_policy(self) -> None:
        full_log = _LOG.replace(
            "INFO x x [graph.py:1] | 7 | 8 | 1 | FULL | 5 |",
            "INFO x x [graph.py:1] | 2 | 2 | 0 | FULL | 1 |\n"
            "INFO x x [graph.py:1] | 3 | 4 | 1 | FULL | 1 |\n"
            "INFO x x [graph.py:1] | 7 | 8 | 1 | FULL | 5 |",
        ).replace(
            "INFO x x [graph.py:1] | 4 | 4 | 0 | FULL | 4 |",
            "INFO x x [graph.py:1] | 1 | 1 | 0 | FULL | 1 |\n"
            "INFO x x [graph.py:1] | 4 | 4 | 0 | FULL | 4 |\n"
            "INFO x x [graph.py:1] | 7 | 8 | 1 | FULL | 1 |",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(3):
                profile = parse_vllm_graph_metrics(
                    full_log.replace("| FULL | 5 |", f"| FULL | {5 + index} |"),
                    profile_id=f"profile-{index}",
                )
                path = root / f"profile-{index}.json"
                path.write_text(json.dumps(profile.artifact_dict()), encoding="utf-8")
                paths.append(path)
            corpus_path = root / "corpus.json"
            promotion_path = root / "promotion.json"
            plan_path = root / "plan.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "build-vllm-graph-corpus",
                            "--corpus-id",
                            "full-corpus",
                            "--train",
                            f"load8={paths[0]}",
                            "--train",
                            f"load32={paths[1]}",
                            "--holdout",
                            f"load96={paths[2]}",
                            "--output",
                            str(corpus_path),
                        ]
                    ),
                    0,
                )
            with redirect_stderr(io.StringIO()):
                status = main(
                    [
                        "plan-vllm-robust-graph-experiment",
                        str(corpus_path),
                        "--campaign-id",
                        "noop-abba",
                        "--blocks",
                        "1",
                        "--pair-seeds",
                        "1",
                        "2",
                        "--promotion-output",
                        str(promotion_path),
                        "--output",
                        str(plan_path),
                    ]
                )

            self.assertEqual(status, 2)
            self.assertFalse(plan_path.exists())
            promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
            self.assertIn("no_policy_change", promotion["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
