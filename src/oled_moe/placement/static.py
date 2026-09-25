from __future__ import annotations

from oled_moe.config import PlacementConfig

from .base import LayerPlacement


class StaticPlacementPolicy:
    """Deterministic placement baseline with no runtime expert migration."""

    def __init__(self, config: PlacementConfig) -> None:
        if config.mode == "gpu_only":
            raise ValueError("gpu_only uses the dynamic GPU cache scheduler")
        self.config = config
        self._layers: dict[tuple[int, int], LayerPlacement] = {}

    def plan(self, *, layer_id: int, num_experts: int) -> LayerPlacement:
        key = (layer_id, num_experts)
        existing = self._layers.get(key)
        if existing is not None:
            return existing

        gpu_count = (
            0
            if self.config.mode == "cpu_only"
            else self.config.static_gpu_experts_per_layer
        )
        if gpu_count > num_experts:
            raise ValueError(
                f"layer {layer_id} has {num_experts} experts but static placement "
                f"requested {gpu_count} GPU experts"
            )
        placement = LayerPlacement(
            gpu_experts=tuple(range(gpu_count)),
            cpu_experts=tuple(range(gpu_count, num_experts)),
        )
        self._layers[key] = placement
        return placement

