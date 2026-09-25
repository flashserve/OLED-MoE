from __future__ import annotations

from collections.abc import Iterable, Mapping

from oled_moe.core import DispatchPlan, ExpertKey


class CooperativeMissScheduler:
    """Split cache misses between H2D/GPU and CPU by current expert priority.

    The split ratio fixes how many misses use each path. ``transfer_high_priority``
    changes only which experts are transferred: high priority first when enabled,
    low priority first when disabled. Expert keys provide deterministic tie breaks.
    """

    def __init__(
        self,
        *,
        cpu_miss_ratio: float,
        transfer_high_priority: bool = True,
        cpu_min_dispatch_experts: int = 1,
    ) -> None:
        if not 0.0 <= cpu_miss_ratio <= 1.0:
            raise ValueError("cpu_miss_ratio must be in [0, 1]")
        if cpu_min_dispatch_experts <= 0:
            raise ValueError("cpu_min_dispatch_experts must be positive")
        self.cpu_miss_ratio = cpu_miss_ratio
        self.transfer_high_priority = transfer_high_priority
        self.cpu_min_dispatch_experts = cpu_min_dispatch_experts
        self._cpu_quota = 0.0

    def plan(
        self,
        *,
        hits: Iterable[ExpertKey],
        misses: Iterable[ExpertKey],
        priorities: Mapping[ExpertKey, float] | None = None,
    ) -> DispatchPlan:
        ordered_hits = tuple(hits)
        ordered_misses = tuple(misses)
        if priorities is not None:
            if self.transfer_high_priority:
                ordered_misses = tuple(
                    sorted(
                        ordered_misses,
                        key=lambda expert: (
                            -priorities.get(expert, 0.0),
                            expert.layer_id,
                            expert.expert_id,
                        ),
                    )
                )
            else:
                ordered_misses = tuple(
                    sorted(
                        ordered_misses,
                        key=lambda expert: (
                            priorities.get(expert, 0.0),
                            expert.layer_id,
                            expert.expert_id,
                        ),
                    )
                )
        miss_count = len(ordered_misses)
        if self.cpu_miss_ratio <= 0.0:
            cpu_count = 0
        elif self.cpu_miss_ratio >= 1.0:
            cpu_count = miss_count
        else:
            # Accumulate a global ratio across eligible layer calls. A minimum
            # dispatch size can then amortize CPU route/copy/thread-pool costs
            # instead of producing one tiny CPU job in almost every layer.
            max_cpu_count = max(0, miss_count - 1)
            if max_cpu_count < self.cpu_min_dispatch_experts:
                cpu_count = 0
            else:
                self._cpu_quota += miss_count * self.cpu_miss_ratio
                requested_cpu_count = int(self._cpu_quota + 1e-12)
                cpu_count = min(max_cpu_count, requested_cpu_count)
                if cpu_count < self.cpu_min_dispatch_experts:
                    cpu_count = 0
                else:
                    self._cpu_quota -= cpu_count
        transfer_count = miss_count - cpu_count
        return DispatchPlan(
            hit_experts=ordered_hits,
            transfer_experts=ordered_misses[:transfer_count],
            cpu_experts=ordered_misses[transfer_count:],
        )
