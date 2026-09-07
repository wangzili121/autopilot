from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from inference_autopilot.evidence import (
    EvidenceGrade,
    EvidenceLedger,
    EvidencePurpose,
    merge_evidence_ledgers,
)
from inference_autopilot.importers import import_results


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class LegacyImporterTest(unittest.TestCase):
    def test_ledger_merge_deduplicates_exact_records_and_rejects_conflicts(self) -> None:
        payload = {
            "workload": {"method": "conditional_is_small_proposal", "requests": 8},
            "runs": [{"completed_qps": 0.4}],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "capacity.json"
            _write(source, payload)
            ledger = import_results(source)

        merged = merge_evidence_ledgers((ledger, ledger))
        self.assertEqual(merged.records, ledger.records)
        conflicting = EvidenceLedger(
            (replace(ledger.records[0], variant="conflicting-variant"),)
        )
        with self.assertRaisesRegex(ValueError, "conflicting evidence records"):
            merge_evidence_ledgers((ledger, conflicting))

    def test_capacity_sweep_is_calibration_only(self) -> None:
        payload = {
            "schema_version": 1,
            "workload": {
                "method": "conditional_is_small_proposal",
                "dataset": "GSM8K",
                "requests": 32,
                "dtype": "float16",
                "base_memory_fraction": 0.54,
                "proposal_memory_fraction": 0.36,
            },
            "runs": [
                {
                    "base_max_num_seqs": 40,
                    "proposal_max_num_seqs": 96,
                    "completed_qps": 0.4,
                    "p95_seconds": 10.0,
                },
                {
                    "base_max_num_seqs": 128,
                    "proposal_max_num_seqs": 768,
                    "completed_qps": 0.8,
                    "p95_seconds": 6.0,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "capacity.json"
            _write(source, payload)
            ledger = import_results(source)

        self.assertEqual(len(ledger.records), 2)
        self.assertFalse(ledger.rejections)
        self.assertTrue(
            all(
                record.quality.grade == EvidenceGrade.B_CONTROLLED_SINGLE
                for record in ledger.records
            )
        )
        self.assertTrue(
            all(
                record.quality.purpose == EvidencePurpose.CALIBRATION_ONLY
                for record in ledger.records
            )
        )
        self.assertEqual(EvidenceLedger.from_dict(ledger.to_dict()), ledger)

    def test_kv_failure_becomes_constraint(self) -> None:
        payload = {
            "models": {"base": "base", "proposal": "proposal"},
            "path": "conditional_is_small_proposal",
            "requests": 96,
            "base_max_num_seqs": 128,
            "proposal_max_num_seqs": 768,
            "experiments": [
                {
                    "name": "shared_2k_apc",
                    "completed_qps": 0.2,
                    "p95_seconds": 400.0,
                    "proposal_kv_peak": 0.12,
                    "preemptions": 0,
                }
            ],
            "unique_8k_failure": {
                "prompt_tokens_approx": 8400,
                "failure": "OOM",
                "temporary_allocation_gib": 4.74,
                "kv_usage_at_failure": 0.02,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "kv.json"
            _write(source, payload)
            ledger = import_results(source)

        self.assertEqual(len(ledger.records), 2)
        failure = next(record for record in ledger.records if "failure" in record.tags)
        self.assertEqual(failure.quality.grade, EvidenceGrade.X_EXCLUDED)
        self.assertEqual(failure.quality.purpose, EvidencePurpose.CONSTRAINT_ONLY)

    def test_directory_scan_records_unknown_and_malformed_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "unknown.json", {"hello": "world"})
            (root / "broken.json").write_text("{", encoding="utf-8")
            ledger = import_results(root)
        self.assertFalse(ledger.records)
        self.assertEqual(len(ledger.rejections), 2)

    def test_directory_scan_does_not_count_copied_artifact_twice(self) -> None:
        payload = {
            "workload": {"method": "conditional_is_small_proposal", "requests": 8},
            "runs": [{"completed_qps": 0.4}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "copy").mkdir()
            _write(root / "original.json", payload)
            _write(root / "copy" / "duplicate.json", payload)
            ledger = import_results(root)
        self.assertEqual(len(ledger.records), 1)
        self.assertEqual(len(ledger.rejections), 1)
        self.assertIn("duplicate content", ledger.rejections[0].reason)

    def test_v5_async_comparison_is_diagnostic(self) -> None:
        payload = {
            "schema_version": 5,
            "algorithm_config": {
                "conditional_is_small_proposal": {
                    "candidate_count": 4,
                    "rollout_count": 2,
                },
                "max_new_tokens": 128,
            },
            "benchmark": "GSM8K cross-request scheduling",
            "environment": {"vllm": "0.18.0"},
            "evaluation": {"problem_indices": [1, 2]},
            "methods": {
                "conditional_is_small_proposal": {
                    "synchronous_seconds": 20.0,
                    "asynchronous_continuous_batching_seconds": 10.0,
                    "wall_time_speedup_synchronous_over_asynchronous": 2.0,
                }
            },
            "models": {"base": {"path": "base"}, "proposal": {"path": "proposal"}},
            "runtime_backend": "vllm",
            "runtime_config": {"vllm": {"base": {"max_num_seqs": 128}}},
            "workers": 8,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy.json"
            _write(source, payload)
            ledger = import_results(source)

        self.assertEqual(len(ledger.records), 1)
        self.assertEqual(ledger.records[0].quality.grade, EvidenceGrade.C_DIAGNOSTIC)
        self.assertIn("weak_baseline_comparison", ledger.records[0].tags)

    def test_approximate_full_run_is_isolated_from_exact_runtime_records(self) -> None:
        payload = {
            "method": "conditional_is_small_proposal",
            "requests": 96,
            "workload": {"dataset": "GSM8K"},
            "runtime": {"base_max_num_seqs": 128, "proposal_max_num_seqs": 768},
            "algorithm": {"rollout_count": 3, "selective_rescoring": True},
            "completed_qps": 0.8,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rescore.json"
            _write(source, payload)
            ledger = import_results(source)

        record = ledger.records[0]
        self.assertEqual(record.algorithm["autopilot_semantic_class"], "approximate_algorithm")
        self.assertIn("approximate_algorithm", record.tags)


if __name__ == "__main__":
    unittest.main()
