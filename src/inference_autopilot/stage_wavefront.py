"""Plan phase-aligned admission waves for staged inference algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any


@dataclass(frozen=True, slots=True)
class StageWavefrontCandidate:
    """One feasible number of synchronous algorithm groups per engine wave."""

    groups_per_wave: int
    sequences_per_wave: int
    wave_count: int
    tail_groups: int
    tail_sequences: int
    exact_partition: bool
    full_wave_utilization: float
    mean_wave_utilization: float
    tail_wave_utilization: float
    balance_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups_per_wave": self.groups_per_wave,
            "sequences_per_wave": self.sequences_per_wave,
            "wave_count": self.wave_count,
            "tail_groups": self.tail_groups,
            "tail_sequences": self.tail_sequences,
            "exact_partition": self.exact_partition,
            "full_wave_utilization": self.full_wave_utilization,
            "mean_wave_utilization": self.mean_wave_utilization,
            "tail_wave_utilization": self.tail_wave_utilization,
            "balance_score": self.balance_score,
        }


@dataclass(frozen=True, slots=True)
class StageWavefrontPlan:
    """A capacity- and workload-bound admission policy."""

    outer_concurrency: int
    sequences_per_group: int
    graph_capture_ceiling: int
    scheduler_sequence_cap: int
    effective_sequence_cap: int
    minimum_full_wave_utilization: float
    selection_class: str
    selected: StageWavefrontCandidate
    candidates: tuple[StageWavefrontCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_concurrency": self.outer_concurrency,
            "sequences_per_group": self.sequences_per_group,
            "graph_capture_ceiling": self.graph_capture_ceiling,
            "scheduler_sequence_cap": self.scheduler_sequence_cap,
            "effective_sequence_cap": self.effective_sequence_cap,
            "minimum_full_wave_utilization": self.minimum_full_wave_utilization,
            "selection_class": self.selection_class,
            "selected": self.selected.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def plan_stage_wavefront(
    *,
    outer_concurrency: int,
    sequences_per_group: int,
    graph_capture_ceiling: int,
    scheduler_sequence_cap: int,
    minimum_full_wave_utilization: float = 0.65,
) -> StageWavefrontPlan:
    """Choose an admission width that preserves graph eligibility and avoids tails.

    A group is one synchronous algorithm call, such as all rollouts for one
    Conditional IS request at one step. Groups remain indivisible so admission
    does not alter algorithm semantics or per-request random streams.
    """

    for name, value in (
        ("outer_concurrency", outer_concurrency),
        ("sequences_per_group", sequences_per_group),
        ("graph_capture_ceiling", graph_capture_ceiling),
        ("scheduler_sequence_cap", scheduler_sequence_cap),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not 0.0 < minimum_full_wave_utilization <= 1.0:
        raise ValueError("minimum_full_wave_utilization must lie in (0, 1]")

    effective_cap = min(graph_capture_ceiling, scheduler_sequence_cap)
    if sequences_per_group > effective_cap:
        raise ValueError(
            "one algorithm group exceeds the graph-eligible scheduler capacity"
        )
    maximum_groups = min(outer_concurrency, effective_cap // sequences_per_group)
    candidates: list[StageWavefrontCandidate] = []
    for groups_per_wave in range(1, maximum_groups + 1):
        wave_count = ceil(outer_concurrency / groups_per_wave)
        remainder = outer_concurrency % groups_per_wave
        tail_groups = remainder or groups_per_wave
        sequences_per_wave = groups_per_wave * sequences_per_group
        tail_sequences = tail_groups * sequences_per_group
        full_utilization = sequences_per_wave / effective_cap
        mean_utilization = (
            outer_concurrency * sequences_per_group / (wave_count * effective_cap)
        )
        tail_utilization = tail_sequences / effective_cap
        tail_shortfall = (
            0.0 if remainder == 0 else (groups_per_wave - remainder) / groups_per_wave
        )
        utilization_shortfall = max(
            0.0, minimum_full_wave_utilization - full_utilization
        )
        balance_score = (
            mean_utilization
            - 0.5 * tail_shortfall
            - 2.0 * utilization_shortfall
            - 0.005 * wave_count
        )
        candidates.append(
            StageWavefrontCandidate(
                groups_per_wave=groups_per_wave,
                sequences_per_wave=sequences_per_wave,
                wave_count=wave_count,
                tail_groups=tail_groups,
                tail_sequences=tail_sequences,
                exact_partition=remainder == 0,
                full_wave_utilization=full_utilization,
                mean_wave_utilization=mean_utilization,
                tail_wave_utilization=tail_utilization,
                balance_score=balance_score,
            )
        )

    exact = [
        candidate
        for candidate in candidates
        if candidate.exact_partition
        and candidate.full_wave_utilization >= minimum_full_wave_utilization
    ]
    if exact:
        selected = max(
            exact,
            key=lambda candidate: (
                candidate.full_wave_utilization,
                -candidate.wave_count,
            ),
        )
        selection_class = "exact_partition_above_minimum_utilization"
    else:
        selected = max(
            candidates,
            key=lambda candidate: (
                candidate.balance_score,
                candidate.mean_wave_utilization,
                candidate.full_wave_utilization,
            ),
        )
        selection_class = "balanced_tail_and_utilization"

    return StageWavefrontPlan(
        outer_concurrency=outer_concurrency,
        sequences_per_group=sequences_per_group,
        graph_capture_ceiling=graph_capture_ceiling,
        scheduler_sequence_cap=scheduler_sequence_cap,
        effective_sequence_cap=effective_cap,
        minimum_full_wave_utilization=minimum_full_wave_utilization,
        selection_class=selection_class,
        selected=selected,
        candidates=tuple(candidates),
    )


__all__ = [
    "StageWavefrontCandidate",
    "StageWavefrontPlan",
    "plan_stage_wavefront",
]
