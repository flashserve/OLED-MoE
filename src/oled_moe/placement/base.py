from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerPlacement:
    """Immutable CPU/GPU ownership for one expert layer."""

    gpu_experts: tuple[int, ...]
    cpu_experts: tuple[int, ...]

    def __post_init__(self) -> None:
        gpu = set(self.gpu_experts)
        cpu = set(self.cpu_experts)
        if gpu.intersection(cpu):
            raise ValueError("CPU and GPU expert placements must be disjoint")

