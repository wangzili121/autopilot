from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from inference_autopilot.calibration.models import canonical_sha256
from inference_autopilot.cli import main
from inference_autopilot.graph_assessment import assess_graph_experiment
from inference_autopilot.graph_experiment import build_graph_experiment_plan
from inference_autopilot.vllm_graph_metrics import parse_vllm_graph_metrics


_PROFILE_LOG = """
[inference-autopilot] graph-stats-begin role=base
INFO 09-04 09:32:40 [cuda_graph.py:123] - Mode: FULL_DECODE_ONLY
INFO 09-04 09:32:40 [cuda_graph.py:123] - Capture sizes: [1, 2, 8]
INFO 09-04 09:32:40 [cuda_graph.py:123] | 1 | 1 | 0 | FULL | 8 |
INFO 09-04 09:32:40 [cuda_graph.py:123] | 7 | 8 | 1 | FULL | 5 |
[inference-autopilot] graph-stats-end role=base
[inference-autopilot] graph-stats-begin role=proposal
INFO 09-04 09:32:40 [cuda_graph.py:123] - Mode: FULL_DECODE_ONLY
INFO 09-04 09:32:40 [cuda_graph.py:123] - Capture sizes: [1, 4, 16]
INFO 09-04 09:32:40 [cuda_graph.py:123] | 4 | 4 | 0 | FULL | 4 |
INFO 09-04 09:32:40 [cuda_graph.py:123] | 15 | 16 | 1 | FULL | 9 |
[inference-autopilot] graph-stats-end role=proposal
"""


def _runtime_log(plan, run, policy) -> str:
    plan_sha256 = canonical_sha256(plan.to_dict())
    engines = {engine.engine_role: engine for engine in policy.engines}
    lines = [
        "[inference-autopilot] "
        f"graph-plan-sha256={plan_sha256} run-id={run.run_id} "
        f"policy-id={policy.policy_id} workload-seed={run.workload_seed}",
        "Graph capturing finished in 7 secs, took 0.30 GiB",
        "init engine (profile, create kv cache, warmup model) took 30.5 seconds",
        "Graph capturing finished in 8 secs, took 0.20 GiB",
        "init engine (profile, create kv cache, warmup model) took 20.5 seconds",
    ]
    for role in ("proposal", "base"):
        engine = engines[role]
        size = engine.capture_sizes[0]
        lines.extend(
            [
                f"[inference-autopilot] graph-stats-begin role={role}",
                f"INFO x y [cuda_graph.py:1] - Mode: {engine.graph_mode}",
                "INFO x y [cuda_graph.py:1] - Capture sizes: "
                + json.dumps(list(engine.capture_sizes)),
                f"INFO x y [cuda_graph.py:1] | {size} | {size} | 0 | FULL | 3 |",
                f"[inference-autopilot] graph-stats-end role={role}",
            ]
        )
    return "\n".join(lines) + "\n"


def _result(plan, run, policy, *, output_suffix: int | None = None) -> dict:
    outputs = [
        {
            "request_index": index,
            "problem_index": 100 + index,
            "gold_answer": str(index),
            "numeric_answer": str(index),
            "token_ids": [run.workload_seed, index, 99],
        }
        for index in range(4)
    ]
    if output_suffix is not None:
        outputs[0]["token_ids"][-1] = output_suffix
    return {
        "method": "conditional_is_small_proposal",
        "dtype": "float16",
        "requests": 4,
        "workers": 4,
        "arrival_qps": 0.0,
        "runtime": {"base_max_num_seqs": 40, "proposal_max_num_seqs": 96},
        "algorithm": {
            "rollout_count": 3,
            "reward": "self_consistency",
            "totals": {"rollout_evaluations_performed": 12},
        },
        "completed_qps": 1.1 if run.variant_role == "candidate" else 1.0,
        "elapsed_seconds": 4.0,
        "latency_seconds": {"mean": 2.0},
        "accuracy": 1.0,
        "compute": {"total_forward_token_slots": 101},
        "outputs": outputs,
        "inference_autopilot_profile": {
            "engine_load_seconds": {"base": 31.0, "proposal": 21.0}
        },
        "graph_experiment": {
            "campaign_id": plan.campaign_id,
            "source_profile_sha256": plan.source_profile_sha256,
            "run": run.to_dict(),
            "policy": policy.to_dict(),
            "policy_sha256": canonical_sha256(policy.to_dict()),
        },
    }


class GraphAssessmentTest(unittest.TestCase):
    def setUp(self) -> None:
        profile = parse_vllm_graph_metrics(_PROFILE_LOG, profile_id="profile")
        self.plan = build_graph_experiment_plan(
            profile,
            campaign_id="graph-abba",
            blocks=1,
            pair_seeds=(11, 22),
        )
        self.comparison_id = "graph-abba--trace-preserving-pruned"

    def _write_runs(self, root: Path, *, mismatch_run_id: str | None = None) -> None:
        policies = {policy.policy_id: policy for policy in self.plan.policies}
        for run in self.plan.runs:
            if run.comparison_id != self.comparison_id:
                continue
            policy = policies[run.policy_id]
            run_root = root / run.run_id
            run_root.mkdir()
            suffix = 17 if run.run_id == mismatch_run_id else None
            (run_root / "result.json").write_text(
                json.dumps(_result(self.plan, run, policy, output_suffix=suffix)),
                encoding="utf-8",
            )
            (run_root / "runner.log").write_text(
                _runtime_log(self.plan, run, policy), encoding="utf-8"
            )
            result_sha256 = hashlib.sha256(
                (run_root / "result.json").read_bytes()
            ).hexdigest()
            log_sha256 = hashlib.sha256(
                (run_root / "runner.log").read_bytes()
            ).hexdigest()
            (run_root / "run.meta").write_text(
                "\n".join(
                    [
                        f"run_id={run.run_id}",
                        "host=test-host",
                        "npu_id=4",
                        "source_repo=/source",
                        "source_revision=abc123",
                        f"source_snapshot_sha256={'1' * 64}",
                        "image=test-image:v1",
                        "requests=4",
                        "workers=4",
                        "config=fixture.toml",
                        f"config_sha256={'2' * 64}",
                        f"dataset_sha256={'3' * 64}",
                        f"wrapper_sha256={'4' * 64}",
                        "graph_plan=plan.json",
                        f"graph_run_id={run.run_id}",
                        f"runner_log_sha256={log_sha256}",
                        f"result_sha256={result_sha256}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

    def test_complete_replayed_pairs_are_formal_when_thresholds_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_runs(root)
            assessment = assess_graph_experiment(
                self.plan,
                root,
                comparison_id=self.comparison_id,
                minimum_requests_per_run=4,
                minimum_blocks=1,
            )

        self.assertTrue(assessment.startup_comparable)
        self.assertTrue(assessment.steady_state_comparable)
        self.assertTrue(assessment.formal_claim_eligible)
        self.assertEqual(assessment.evidence_tier, "formal_paired")
        self.assertAlmostEqual(
            assessment.summary["comparable_pairs_qps_geomean_ratio"], 1.1
        )
        self.assertEqual(
            assessment.summary["median_wrapper_load_seconds_delta"], 0.0
        )
        self.assertEqual(
            assessment.runs[0].startup["base"]["wrapper_load_seconds"], 31.0
        )
        self.assertTrue(assessment.runs[0].provenance_complete)
        self.assertTrue(all(pair.environment_exact for pair in assessment.pairs))

    def test_token_replay_mismatch_blocks_steady_state_claim(self) -> None:
        candidate = next(
            run
            for run in self.plan.runs
            if run.comparison_id == self.comparison_id
            and run.variant_role == "candidate"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_runs(root, mismatch_run_id=candidate.run_id)
            assessment = assess_graph_experiment(
                self.plan,
                root,
                comparison_id=self.comparison_id,
                minimum_requests_per_run=4,
                minimum_blocks=1,
            )

        self.assertTrue(assessment.startup_comparable)
        self.assertFalse(assessment.steady_state_comparable)
        self.assertIn(
            "stochastic_replay_mismatch", {issue.code for issue in assessment.issues}
        )
        mismatched_pair = next(
            pair for pair in assessment.pairs if not pair.outputs_exact
        )
        self.assertEqual(mismatched_pair.compared_request_count, 4)
        self.assertEqual(mismatched_pair.exact_output_count, 3)
        self.assertEqual(mismatched_pair.semantic_answer_match_count, 4)
        self.assertEqual(mismatched_pair.accuracy_candidate_minus_baseline, 0.0)

    def test_log_policy_tampering_invalidates_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_runs(root)
            run = next(
                run for run in self.plan.runs if run.comparison_id == self.comparison_id
            )
            log_path = root / run.run_id / "runner.log"
            log_path.write_text(
                log_path.read_text().replace("Capture sizes: [1, 2, 8]", "Capture sizes: [1, 8]"),
                encoding="utf-8",
            )
            assessment = assess_graph_experiment(
                self.plan,
                root,
                comparison_id=self.comparison_id,
                minimum_requests_per_run=4,
                minimum_blocks=1,
            )

        self.assertFalse(assessment.startup_comparable)
        self.assertEqual(assessment.valid_run_count, 3)
        self.assertIn("invalid_run_artifact", {issue.code for issue in assessment.issues})

    def test_cli_writes_content_addressed_diagnostic(self) -> None:
        candidate = next(
            run
            for run in self.plan.runs
            if run.comparison_id == self.comparison_id
            and run.variant_role == "candidate"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_runs(root, mismatch_run_id=candidate.run_id)
            plan_path = root / "plan.json"
            output = root / "assessment.json"
            plan_path.write_text(json.dumps(self.plan.artifact_dict()), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "assess-vllm-graph-experiment",
                        str(plan_path),
                        str(root),
                        "--comparison-id",
                        self.comparison_id,
                        "--minimum-requests",
                        "4",
                        "--minimum-blocks",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
            artifact = json.loads(output.read_text(encoding="utf-8"))
            payload = deepcopy(artifact)
            digest = payload.pop("graph_experiment_assessment_sha256")

        self.assertEqual(status, 2)
        self.assertEqual(digest, canonical_sha256(payload))


if __name__ == "__main__":
    unittest.main()
