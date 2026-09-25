from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, order=True)
class ExpertKey:
    layer_id: int
    expert_id: int


@dataclass(frozen=True)
class RouteBatch:
    expert_ids: torch.Tensor
    gate_weights: torch.Tensor
    previous_confidence: torch.Tensor | None = None


@dataclass(frozen=True)
class DispatchPlan:
    hit_experts: tuple[ExpertKey, ...]
    transfer_experts: tuple[ExpertKey, ...]
    cpu_experts: tuple[ExpertKey, ...] = ()

    def __post_init__(self) -> None:
        hit = set(self.hit_experts)
        transfer = set(self.transfer_experts)
        cpu = set(self.cpu_experts)
        if hit.intersection(transfer) or hit.intersection(cpu) or transfer.intersection(cpu):
            raise ValueError("dispatch plan expert partitions must be disjoint")
