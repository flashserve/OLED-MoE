from __future__ import annotations

from collections.abc import Iterable, Mapping

from oled_moe.core import DispatchPlan, ExpertKey


class GPUOnlyMissScheduler:
    """Correctness baseline: every cache miss is transferred and run on GPU."""

    def plan(
        self,
        *,
        hits: Iterable[ExpertKey],
        misses: Iterable[ExpertKey],
        priorities: Mapping[ExpertKey, float] | None = None,
    ) -> DispatchPlan:
        return DispatchPlan(
            hit_experts=tuple(hits),
            transfer_experts=tuple(misses),
            cpu_experts=(),
        )
