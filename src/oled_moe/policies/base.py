from __future__ import annotations

from typing import Protocol

from oled_moe.core import ExpertKey, RouteBatch


class ExpertPriorityPolicy(Protocol):
    def score(self, layer_id: int, route: RouteBatch) -> dict[ExpertKey, float]: ...


class LayerCapacityPolicy(Protocol):
    def capacities(self) -> dict[int, int]: ...
