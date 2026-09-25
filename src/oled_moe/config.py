from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CachePolicy = Literal["lru", "lru_observed", "lfu", "priority"]
ExecutionMode = Literal["gpu_only", "cpu_only", "static_gpu", "static_hybrid", "cooperative"]
CPUExecutionStrategy = Literal["serial", "expert_parallel"]
CPUPointwiseMode = Literal["baseline", "inplace"]


@dataclass(frozen=True)
class CacheConfig:
    """Bounded per-layer GPU expert-cache configuration for the first milestone."""

    capacity_per_layer: int = 100
    policy: CachePolicy = "lru"
    confidence_power: float = 1.0
    gate_power: float = 1.0
    update_interval: int = 1
    selective_priority: bool = False
    priority_layer_fraction: float = 0.5
    utilization_ema_decay: float = 0.9
    min_priority_opportunity: float = 0.02
    priority_hysteresis: float = 0.01

    def __post_init__(self) -> None:
        if self.capacity_per_layer <= 0:
            raise ValueError("capacity_per_layer must be positive")
        if self.policy not in {"lru", "lru_observed", "lfu", "priority"}:
            raise ValueError(f"unsupported cache policy: {self.policy}")
        if self.confidence_power <= 0:
            raise ValueError("confidence_power must be positive")
        if self.gate_power <= 0:
            raise ValueError("gate_power must be positive")
        if self.update_interval < 0:
            raise ValueError("update_interval must be non-negative")
        if not 0.0 < self.priority_layer_fraction <= 1.0:
            raise ValueError("priority_layer_fraction must be in (0, 1]")
        if not 0.0 <= self.utilization_ema_decay < 1.0:
            raise ValueError("utilization_ema_decay must be in [0, 1)")
        if not 0.0 <= self.min_priority_opportunity <= 1.0:
            raise ValueError("min_priority_opportunity must be in [0, 1]")
        if not 0.0 <= self.priority_hysteresis <= 1.0:
            raise ValueError("priority_hysteresis must be in [0, 1]")


@dataclass(frozen=True)
class LayerCapacityConfig:
    """Layer-heterogeneous allocation over a fixed global slot budget."""

    enabled: bool = False
    apply: bool = False
    warmup_forwards: int = 8
    update_interval: int = 8
    steady_update_interval: int = 30
    convergence_patience: int = 3
    convergence_moved_ratio: float = 0.005
    ema_decay: float = 0.8
    activation_power: float = 1.0
    utilization_power: float = 1.0
    utilization_floor: float = 0.05
    min_capacity_ratio: float = 0.25
    max_capacity_ratio: float = 2.0
    max_slots_moved_ratio: float = 0.10
    capacity_hysteresis: float = 0.01

    def __post_init__(self) -> None:
        if self.apply and not self.enabled:
            raise ValueError("layer capacity apply requires enabled=True")
        if self.warmup_forwards <= 0:
            raise ValueError("layer capacity warmup_forwards must be positive")
        if self.update_interval <= 0:
            raise ValueError("layer capacity update_interval must be positive")
        if self.steady_update_interval < self.update_interval:
            raise ValueError(
                "layer capacity steady_update_interval must be at least update_interval"
            )
        if self.convergence_patience <= 0:
            raise ValueError("layer capacity convergence_patience must be positive")
        if not 0.0 < self.convergence_moved_ratio <= 1.0:
            raise ValueError("layer capacity convergence_moved_ratio must be in (0, 1]")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("layer capacity ema_decay must be in [0, 1)")
        if self.activation_power <= 0 or self.utilization_power <= 0:
            raise ValueError("layer capacity powers must be positive")
        if self.utilization_floor < 0:
            raise ValueError("layer capacity utilization_floor must be non-negative")
        if not 0.0 < self.min_capacity_ratio <= 1.0:
            raise ValueError("layer capacity min_capacity_ratio must be in (0, 1]")
        if self.max_capacity_ratio < 1.0:
            raise ValueError("layer capacity max_capacity_ratio must be at least 1")
        if not 0.0 < self.max_slots_moved_ratio <= 1.0:
            raise ValueError("layer capacity max_slots_moved_ratio must be in (0, 1]")
        if self.capacity_hysteresis < 0:
            raise ValueError("layer capacity capacity_hysteresis must be non-negative")


@dataclass(frozen=True)
class TransferConfig:
    """CPU-to-GPU transfer resources."""

    use_copy_stream: bool = True
    overlap_compute: bool = True
    coalesce_expert_weights: bool = True
    gpu_expert_batch_size: int = 64

    def __post_init__(self) -> None:
        if self.gpu_expert_batch_size < 2:
            raise ValueError("gpu_expert_batch_size must be at least two")


@dataclass(frozen=True)
class ProfilingConfig:
    """Opt-in detailed timing; CUDA events add overhead and synchronization."""

    enabled: bool = False


@dataclass(frozen=True)
class PlacementConfig:
    """Expert placement used by the correctness and static baselines."""

    mode: ExecutionMode = "gpu_only"
    static_gpu_experts_per_layer: int = 0
    cpu_miss_ratio: float = 0.5
    cpu_min_dispatch_experts: int = 1
    transfer_high_priority: bool = True

    def __post_init__(self) -> None:
        supported = {"gpu_only", "cpu_only", "static_gpu", "static_hybrid", "cooperative"}
        if self.mode not in supported:
            raise ValueError(f"unsupported execution mode: {self.mode}")
        if self.static_gpu_experts_per_layer < 0:
            raise ValueError("static_gpu_experts_per_layer must be non-negative")
        static_modes = {"static_gpu", "static_hybrid"}
        if self.mode in static_modes and self.static_gpu_experts_per_layer <= 0:
            raise ValueError(f"{self.mode} requires at least one static GPU expert")
        if self.mode not in static_modes and self.static_gpu_experts_per_layer != 0:
            raise ValueError("static_gpu_experts_per_layer is only valid for static GPU modes")
        if not 0.0 <= self.cpu_miss_ratio <= 1.0:
            raise ValueError("cpu_miss_ratio must be in [0, 1]")
        if self.cpu_min_dispatch_experts <= 0:
            raise ValueError("cpu_min_dispatch_experts must be positive")


@dataclass(frozen=True)
class CPUConfig:
    """Portable PyTorch CPU expert backend configuration."""

    num_threads: int = 16
    strategy: CPUExecutionStrategy = "expert_parallel"
    pointwise_mode: CPUPointwiseMode = "inplace"
    compact_io: bool = True

    def __post_init__(self) -> None:
        if self.num_threads <= 0:
            raise ValueError("cpu num_threads must be positive")
        if self.strategy not in {"serial", "expert_parallel"}:
            raise ValueError(f"unsupported CPU execution strategy: {self.strategy}")
        if self.pointwise_mode not in {"baseline", "inplace"}:
            raise ValueError(f"unsupported CPU pointwise mode: {self.pointwise_mode}")


@dataclass(frozen=True)
class OLEDConfig:
    """Top-level OLED runtime configuration."""

    cache: CacheConfig = field(default_factory=CacheConfig)
    layer_capacity: LayerCapacityConfig = field(default_factory=LayerCapacityConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    cpu: CPUConfig = field(default_factory=CPUConfig)
    pin_memory: bool = True
