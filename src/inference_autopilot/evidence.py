"""Evidence ledger contracts used before measurements reach an optimizer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class EvidenceGrade(str, Enum):
    A_FORMAL_PAIRED = "A_formal_paired"
    B_CONTROLLED_SINGLE = "B_controlled_single"
    C_DIAGNOSTIC = "C_diagnostic"
    X_EXCLUDED = "X_excluded"


class EvidencePurpose(str, Enum):
    SELECTOR_FIT_AND_CLAIM = "selector_fit_and_claim"
    CALIBRATION_ONLY = "calibration_only"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    CONSTRAINT_ONLY = "constraint_only"


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    path: str
    sha256: str
    imported_format: str
    locator: str

    def __post_init__(self) -> None:
        if not self.path or not self.imported_format or not self.locator:
            raise ValueError("source artifact fields cannot be empty")
        invalid_character = any(
            character not in "0123456789abcdef" for character in self.sha256
        )
        if len(self.sha256) != 64 or invalid_character:
            raise ValueError("source artifact sha256 must be a lowercase hexadecimal digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "imported_format": self.imported_format,
            "locator": self.locator,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SourceArtifact":
        keys = {"path", "sha256", "imported_format", "locator"}
        _expect_exact_keys(raw, keys, "source artifact")
        return cls(**{key: str(raw[key]) for key in keys})


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    grade: EvidenceGrade
    purpose: EvidencePurpose
    claim_eligible: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reasons or any(not reason for reason in self.reasons):
            raise ValueError("quality assessment requires at least one reason")
        expected_purpose = {
            EvidenceGrade.A_FORMAL_PAIRED: EvidencePurpose.SELECTOR_FIT_AND_CLAIM,
            EvidenceGrade.B_CONTROLLED_SINGLE: EvidencePurpose.CALIBRATION_ONLY,
            EvidenceGrade.C_DIAGNOSTIC: EvidencePurpose.DIAGNOSTIC_ONLY,
            EvidenceGrade.X_EXCLUDED: EvidencePurpose.CONSTRAINT_ONLY,
        }[self.grade]
        if self.purpose != expected_purpose:
            raise ValueError(f"grade {self.grade.value} requires purpose {expected_purpose.value}")
        expected_claim_eligibility = self.grade == EvidenceGrade.A_FORMAL_PAIRED
        if self.claim_eligible != expected_claim_eligibility:
            raise ValueError("only grade-A evidence can be claim eligible")

    def to_dict(self) -> dict[str, Any]:
        return {
            "grade": self.grade.value,
            "purpose": self.purpose.value,
            "claim_eligible": self.claim_eligible,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "QualityAssessment":
        keys = {"grade", "purpose", "claim_eligible", "reasons"}
        _expect_exact_keys(raw, keys, "quality assessment")
        reasons = raw["reasons"]
        if not isinstance(reasons, list):
            raise ValueError("quality assessment reasons must be a list")
        if not isinstance(raw["claim_eligible"], bool):
            raise ValueError("quality assessment claim_eligible must be a boolean")
        return cls(
            grade=EvidenceGrade(str(raw["grade"])),
            purpose=EvidencePurpose(str(raw["purpose"])),
            claim_eligible=raw["claim_eligible"],
            reasons=tuple(str(reason) for reason in reasons),
        )


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    record_id: str
    campaign: str
    variant: str
    source: SourceArtifact
    workload: Mapping[str, Any]
    configuration: Mapping[str, Any]
    algorithm: Mapping[str, Any]
    environment: Mapping[str, Any]
    metrics: Mapping[str, Any]
    quality: QualityAssessment
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.record_id or not self.campaign or not self.variant:
            raise ValueError("record identity fields cannot be empty")
        for name in ("workload", "configuration", "algorithm", "environment", "metrics"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"record {name} must be an object")
        if not self.metrics:
            raise ValueError("an evidence record must contain metrics")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("evidence record tags must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "campaign": self.campaign,
            "variant": self.variant,
            "source": self.source.to_dict(),
            "workload": dict(self.workload),
            "configuration": dict(self.configuration),
            "algorithm": dict(self.algorithm),
            "environment": dict(self.environment),
            "metrics": dict(self.metrics),
            "quality": self.quality.to_dict(),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvidenceRecord":
        keys = {
            "record_id",
            "campaign",
            "variant",
            "source",
            "workload",
            "configuration",
            "algorithm",
            "environment",
            "metrics",
            "quality",
            "tags",
        }
        _expect_exact_keys(raw, keys, "evidence record")
        object_fields = (
            "source",
            "workload",
            "configuration",
            "algorithm",
            "environment",
            "metrics",
            "quality",
        )
        for name in object_fields:
            if not isinstance(raw[name], Mapping):
                raise ValueError(f"evidence record {name} must be an object")
        if not isinstance(raw["tags"], list):
            raise ValueError("evidence record tags must be a list")
        return cls(
            record_id=str(raw["record_id"]),
            campaign=str(raw["campaign"]),
            variant=str(raw["variant"]),
            source=SourceArtifact.from_dict(raw["source"]),
            workload=dict(raw["workload"]),
            configuration=dict(raw["configuration"]),
            algorithm=dict(raw["algorithm"]),
            environment=dict(raw["environment"]),
            metrics=dict(raw["metrics"]),
            quality=QualityAssessment.from_dict(raw["quality"]),
            tags=tuple(str(tag) for tag in raw["tags"]),
        )


@dataclass(frozen=True, slots=True)
class ImportRejection:
    path: str
    sha256: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "reason": self.reason}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ImportRejection":
        keys = {"path", "sha256", "reason"}
        _expect_exact_keys(raw, keys, "import rejection")
        sha256 = raw["sha256"]
        return cls(
            path=str(raw["path"]),
            sha256=None if sha256 is None else str(sha256),
            reason=str(raw["reason"]),
        )


@dataclass(frozen=True, slots=True)
class EvidenceLedger:
    records: tuple[EvidenceRecord, ...]
    rejections: tuple[ImportRejection, ...] = ()
    schema_version: str = "1.0"
    producer: str = "inference-autopilot/0.1.0"

    def __post_init__(self) -> None:
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("evidence ledger record ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "records": [record.to_dict() for record in self.records],
            "rejections": [rejection.to_dict() for rejection in self.rejections],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvidenceLedger":
        keys = {"schema_version", "producer", "records", "rejections"}
        _expect_exact_keys(raw, keys, "evidence ledger")
        if raw["schema_version"] != "1.0":
            raise ValueError(f"unsupported evidence ledger schema: {raw['schema_version']}")
        if not isinstance(raw["records"], list) or not isinstance(raw["rejections"], list):
            raise ValueError("evidence ledger records and rejections must be lists")
        return cls(
            records=tuple(EvidenceRecord.from_dict(item) for item in raw["records"]),
            rejections=tuple(ImportRejection.from_dict(item) for item in raw["rejections"]),
            schema_version=str(raw["schema_version"]),
            producer=str(raw["producer"]),
        )

    def audit(self) -> dict[str, Any]:
        grades = Counter(record.quality.grade.value for record in self.records)
        purposes = Counter(record.quality.purpose.value for record in self.records)
        formats = Counter(record.source.imported_format for record in self.records)
        return {
            "record_count": len(self.records),
            "rejection_count": len(self.rejections),
            "claim_eligible_count": sum(record.quality.claim_eligible for record in self.records),
            "by_grade": dict(sorted(grades.items())),
            "by_purpose": dict(sorted(purposes.items())),
            "by_imported_format": dict(sorted(formats.items())),
        }


def merge_evidence_ledgers(ledgers: tuple[EvidenceLedger, ...]) -> EvidenceLedger:
    """Merge compatible ledgers deterministically without hiding conflicts."""

    if not ledgers:
        raise ValueError("at least one evidence ledger is required")
    producers = {ledger.producer for ledger in ledgers}
    if len(producers) != 1:
        raise ValueError("evidence ledger producers must match")

    records_by_id: dict[str, EvidenceRecord] = {}
    for ledger in ledgers:
        for record in ledger.records:
            existing = records_by_id.get(record.record_id)
            if existing is not None and existing.to_dict() != record.to_dict():
                raise ValueError(
                    f"conflicting evidence records share id {record.record_id}"
                )
            records_by_id[record.record_id] = record

    rejections_by_value: dict[tuple[str, str | None, str], ImportRejection] = {}
    for ledger in ledgers:
        for rejection in ledger.rejections:
            key = (rejection.path, rejection.sha256, rejection.reason)
            rejections_by_value[key] = rejection

    return EvidenceLedger(
        records=tuple(records_by_id[key] for key in sorted(records_by_id)),
        rejections=tuple(
            rejections_by_value[key]
            for key in sorted(
                rejections_by_value,
                key=lambda item: (item[0], item[1] or "", item[2]),
            )
        ),
        producer=next(iter(producers)),
    )
