from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from inference_autopilot.attempt_ledger import (
    append_attempt,
    audit_attempt_ledger,
    load_attempt_ledger,
)


def _canonical_sha256(value: object) -> str:
    import hashlib

    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _hashed(payload: dict[str, object], field: str) -> dict[str, object]:
    return {**payload, field: _canonical_sha256(payload)}


class AttemptLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.campaign = Path(self.temporary.name)
        self.run = {
            "run_id": "run-a",
            "sequence_index": 0,
            "configuration_id": "config-a",
        }
        self.plan = {
            "schema_version": "1.0",
            "runs": [self.run],
        }
        self.plan_path = self.campaign / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        self.ledger = self.campaign / "attempt-ledger.jsonl"
        self.admission = self.campaign / "admission" / "window-000.report.json"
        self.admission.parent.mkdir()
        self.admission.write_text(
            json.dumps(
                _hashed(
                    {"schema_version": "1.0", "status": "clean"},
                    "report_sha256",
                )
            ),
            encoding="utf-8",
        )

    def _bundle(self, relative: str, attempt: int, outcome: str) -> Path:
        bundle = self.campaign / relative
        bundle.mkdir(parents=True)
        manifest = _hashed(
            {
                "schema_version": "1.0",
                "plan_sha256": _canonical_sha256(self.plan),
                "run": self.run,
            },
            "run_manifest_sha256",
        )
        (bundle / "run-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        run_outcome = "success" if outcome == "accepted" else outcome
        (bundle / "execution.meta").write_text(
            "\n".join(
                (
                    "run_id=run-a",
                    "logical_run_id=run-a",
                    f"attempt_id=attempt-{attempt:03d}",
                    f"run_outcome={run_outcome}",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        host_status = "clean" if outcome == "accepted" else "contaminated"
        (bundle / "host-interference-report.json").write_text(
            json.dumps(
                _hashed(
                    {"schema_version": "1.0", "status": host_status},
                    "report_sha256",
                )
            ),
            encoding="utf-8",
        )
        if outcome == "accepted":
            (bundle / "observation.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-a",
                        "run_manifest_sha256": manifest["run_manifest_sha256"],
                        "status": "success",
                    }
                ),
                encoding="utf-8",
            )
        return bundle

    def test_contaminated_attempt_then_accepted_attempt_audits(self) -> None:
        contaminated = self._bundle(
            "attempts/run-a/attempt-000", 0, "environment_contaminated"
        )
        accepted = self._bundle("run-a", 1, "accepted")

        first = append_attempt(
            self.ledger,
            campaign_dir=self.campaign,
            bundle_dir=contaminated,
            logical_run_id="run-a",
            attempt_index=0,
            outcome="environment_contaminated",
            admission_report=self.admission,
        )
        second = append_attempt(
            self.ledger,
            campaign_dir=self.campaign,
            bundle_dir=accepted,
            logical_run_id="run-a",
            attempt_index=1,
            outcome="accepted",
            admission_report=self.admission,
        )
        report = audit_attempt_ledger(
            self.ledger,
            campaign_dir=self.campaign,
            plan_path=self.plan_path,
        )

        self.assertEqual(second["previous_record_sha256"], first["record_sha256"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["attempt_count"], 2)
        self.assertEqual(report["contaminated_attempt_count"], 1)

    def test_ledger_hash_chain_rejects_tampering(self) -> None:
        bundle = self._bundle("run-a", 0, "accepted")
        append_attempt(
            self.ledger,
            campaign_dir=self.campaign,
            bundle_dir=bundle,
            logical_run_id="run-a",
            attempt_index=0,
            outcome="accepted",
            admission_report=self.admission,
        )
        record = json.loads(self.ledger.read_text(encoding="utf-8"))
        record["outcome"] = "run_failed"
        self.ledger.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_attempt_ledger(self.ledger)

    def test_audit_rejects_artifact_mutation(self) -> None:
        bundle = self._bundle("run-a", 0, "accepted")
        append_attempt(
            self.ledger,
            campaign_dir=self.campaign,
            bundle_dir=bundle,
            logical_run_id="run-a",
            attempt_index=0,
            outcome="accepted",
            admission_report=self.admission,
        )
        with (bundle / "execution.meta").open("a", encoding="utf-8") as stream:
            stream.write("mutated=true\n")

        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            audit_attempt_ledger(
                self.ledger,
                campaign_dir=self.campaign,
                plan_path=self.plan_path,
            )

    def test_attempt_indices_are_contiguous(self) -> None:
        bundle = self._bundle(
            "attempts/run-a/attempt-001", 1, "environment_contaminated"
        )
        with self.assertRaisesRegex(ValueError, "must be 0"):
            append_attempt(
                self.ledger,
                campaign_dir=self.campaign,
                bundle_dir=bundle,
                logical_run_id="run-a",
                attempt_index=1,
                outcome="environment_contaminated",
                admission_report=self.admission,
            )

    def test_partial_audit_identifies_resumable_logical_run(self) -> None:
        contaminated = self._bundle(
            "attempts/run-a/attempt-000", 0, "environment_contaminated"
        )
        append_attempt(
            self.ledger,
            campaign_dir=self.campaign,
            bundle_dir=contaminated,
            logical_run_id="run-a",
            attempt_index=0,
            outcome="environment_contaminated",
            admission_report=self.admission,
        )

        report = audit_attempt_ledger(
            self.ledger,
            campaign_dir=self.campaign,
            plan_path=self.plan_path,
            require_complete=False,
        )

        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["active_run_id"], "run-a")
        self.assertEqual(report["missing_run_ids"], ["run-a"])
        with self.assertRaisesRegex(ValueError, "no accepted attempt"):
            audit_attempt_ledger(
                self.ledger,
                campaign_dir=self.campaign,
                plan_path=self.plan_path,
            )


if __name__ == "__main__":
    unittest.main()
