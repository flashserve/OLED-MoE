from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from oled_moe.config import LayerCapacityConfig


@dataclass
class _LayerWindow:
    num_experts: int
    initial_capacity: int
    recommended_capacity: int
    applied_capacity: int
    active_experts: int = 0
    hit_experts: int = 0
    slot_observations: int = 0
    calls: int = 0
    activation_ratio_ema: float = 0.0
    cache_utilization_ema: float = 0.0
    allocation_score: float = 0.0
    ema_initialized: bool = False

    def clear_window(self) -> None:
        self.active_experts = 0
        self.hit_experts = 0
        self.slot_observations = 0
        self.calls = 0


class LayerCapacityController:
    """Online layer-capacity controller for a fixed global expert-slot budget."""

    def __init__(self, config: LayerCapacityConfig) -> None:
        self.config = config
        self._layers: dict[int, _LayerWindow] = {}
        self._seen_layers: set[int] = set()
        self.completed_forwards = 0
        self.update_count = 0
        self.last_moved_slots = 0
        self.stable_update_count = 0
        self.converged = False
        self.current_update_interval = config.update_interval
        self.next_update_forward = config.warmup_forwards
        self.history: list[dict[str, Any]] = []

    @property
    def total_slots(self) -> int:
        return sum(layer.initial_capacity for layer in self._layers.values())

    def register_layer(
        self,
        *,
        layer_id: int,
        num_experts: int,
        initial_capacity: int,
    ) -> None:
        capacity = min(num_experts, initial_capacity)
        existing = self._layers.get(layer_id)
        if existing is not None:
            if existing.num_experts != num_experts or existing.initial_capacity != capacity:
                raise ValueError(f"layer {layer_id} capacity metadata changed")
            return
        self._layers[layer_id] = _LayerWindow(
            num_experts=num_experts,
            initial_capacity=capacity,
            recommended_capacity=capacity,
            applied_capacity=capacity,
        )

    def reset(self, *, initial_capacity: int | None = None) -> None:
        for layer in self._layers.values():
            capacity = (
                layer.initial_capacity
                if initial_capacity is None
                else min(layer.num_experts, initial_capacity)
            )
            layer.initial_capacity = capacity
            layer.recommended_capacity = capacity
            layer.applied_capacity = capacity
            layer.activation_ratio_ema = 0.0
            layer.cache_utilization_ema = 0.0
            layer.allocation_score = 0.0
            layer.ema_initialized = False
            layer.clear_window()
        self._seen_layers.clear()
        self.completed_forwards = 0
        self.update_count = 0
        self.last_moved_slots = 0
        self.stable_update_count = 0
        self.converged = False
        self.current_update_interval = self.config.update_interval
        self.next_update_forward = self.config.warmup_forwards
        self.history.clear()

    def observe(
        self,
        *,
        layer_id: int,
        active_experts: int,
        hit_experts: int,
        capacity: int,
    ) -> bool:
        if not self.config.enabled:
            return False
        layer = self._layers[layer_id]
        if layer_id in self._seen_layers:
            # A partial/nonstandard traversal cannot form a comparable complete
            # transformer-forward window. Start a fresh traversal at this layer.
            self._seen_layers.clear()
        layer.active_experts += active_experts
        layer.hit_experts += hit_experts
        layer.slot_observations += capacity
        layer.calls += 1
        self._seen_layers.add(layer_id)
        if self._seen_layers != set(self._layers):
            return False
        self._seen_layers.clear()
        self.completed_forwards += 1
        if self.completed_forwards < self.next_update_forward:
            return False
        self._update_recommendation()
        self._update_schedule()
        return True

    def _update_recommendation(self) -> None:
        scores: dict[int, float] = {}
        minimums: dict[int, int] = {}
        maximums: dict[int, int] = {}
        for layer_id, layer in self._layers.items():
            activation_sample = (
                layer.active_experts / (layer.calls * layer.num_experts) if layer.calls else 0.0
            )
            utilization_sample = (
                layer.hit_experts / layer.slot_observations if layer.slot_observations else 0.0
            )
            if layer.ema_initialized:
                decay = self.config.ema_decay
                layer.activation_ratio_ema = (
                    decay * layer.activation_ratio_ema + (1.0 - decay) * activation_sample
                )
                layer.cache_utilization_ema = (
                    decay * layer.cache_utilization_ema + (1.0 - decay) * utilization_sample
                )
            else:
                layer.activation_ratio_ema = activation_sample
                layer.cache_utilization_ema = utilization_sample
                layer.ema_initialized = True
            layer.allocation_score = (
                layer.activation_ratio_ema + 1e-6
            ) ** self.config.activation_power * (
                self.config.utilization_floor + layer.cache_utilization_ema + 1e-6
            ) ** self.config.utilization_power
            scores[layer_id] = layer.allocation_score
            minimums[layer_id] = max(
                1,
                min(
                    layer.num_experts,
                    math.floor(layer.initial_capacity * self.config.min_capacity_ratio),
                ),
            )
            maximums[layer_id] = min(
                layer.num_experts,
                max(
                    layer.initial_capacity,
                    math.ceil(layer.initial_capacity * self.config.max_capacity_ratio),
                ),
            )

        desired = self._bounded_apportion(
            scores=scores,
            budget=self.total_slots,
            minimums=minimums,
            maximums=maximums,
        )
        current = {layer_id: layer.recommended_capacity for layer_id, layer in self._layers.items()}
        recommended, moved = self._limit_moves(
            current=current,
            desired=desired,
            scores=scores,
        )
        for layer_id, layer in self._layers.items():
            layer.recommended_capacity = recommended[layer_id]
            layer.clear_window()
        self.update_count += 1
        self.last_moved_slots = moved
        self.history.append(
            {
                "completed_forwards": self.completed_forwards,
                "moved_slots": moved,
                "capacities": dict(sorted(recommended.items())),
            }
        )

    def _update_schedule(self) -> None:
        convergence_limit = max(
            1,
            math.floor(self.total_slots * self.config.convergence_moved_ratio),
        )
        if self.last_moved_slots <= convergence_limit:
            self.stable_update_count += 1
        else:
            self.stable_update_count = 0
            self.converged = False
        if self.stable_update_count >= self.config.convergence_patience:
            self.converged = True
        self.current_update_interval = (
            self.config.steady_update_interval if self.converged else self.config.update_interval
        )
        self.next_update_forward = self.completed_forwards + self.current_update_interval
        self.history[-1].update(
            {
                "convergence_limit": convergence_limit,
                "stable_updates": self.stable_update_count,
                "converged": self.converged,
                "next_update_interval": self.current_update_interval,
                "next_update_forward": self.next_update_forward,
            }
        )

    @staticmethod
    def _bounded_apportion(
        *,
        scores: dict[int, float],
        budget: int,
        minimums: dict[int, int],
        maximums: dict[int, int],
    ) -> dict[int, int]:
        if sum(minimums.values()) > budget or sum(maximums.values()) < budget:
            raise ValueError("layer-capacity bounds cannot satisfy the slot budget")
        allocation = dict(minimums)
        remaining = budget - sum(allocation.values())
        while remaining:
            eligible = [
                layer_id for layer_id in allocation if allocation[layer_id] < maximums[layer_id]
            ]
            if not eligible:
                raise AssertionError("slot apportionment exhausted eligible layers")
            weights = {layer_id: max(0.0, scores.get(layer_id, 0.0)) for layer_id in eligible}
            weight_sum = sum(weights.values())
            if weight_sum == 0.0:
                weights = {layer_id: 1.0 for layer_id in eligible}
                weight_sum = float(len(eligible))
            ideals = {layer_id: remaining * weights[layer_id] / weight_sum for layer_id in eligible}
            grants = {
                layer_id: min(
                    maximums[layer_id] - allocation[layer_id],
                    math.floor(ideals[layer_id]),
                )
                for layer_id in eligible
            }
            granted = sum(grants.values())
            if granted:
                for layer_id, grant in grants.items():
                    allocation[layer_id] += grant
                remaining -= granted
                continue
            ranked = sorted(
                eligible,
                key=lambda layer_id: (
                    ideals[layer_id] - math.floor(ideals[layer_id]),
                    weights[layer_id],
                    -layer_id,
                ),
                reverse=True,
            )
            for layer_id in ranked:
                if not remaining:
                    break
                if allocation[layer_id] < maximums[layer_id]:
                    allocation[layer_id] += 1
                    remaining -= 1
        return allocation

    def _limit_moves(
        self,
        *,
        current: dict[int, int],
        desired: dict[int, int],
        scores: dict[int, float],
    ) -> tuple[dict[int, int], int]:
        result = dict(current)
        move_limit = max(
            1,
            math.floor(self.total_slots * self.config.max_slots_moved_ratio),
        )
        receivers = sorted(
            (layer_id for layer_id in result if desired[layer_id] > result[layer_id]),
            key=lambda layer_id: (scores[layer_id], -layer_id),
            reverse=True,
        )
        donors = sorted(
            (layer_id for layer_id in result if desired[layer_id] < result[layer_id]),
            key=lambda layer_id: (scores[layer_id], layer_id),
        )
        moved = 0
        receiver_index = 0
        donor_index = 0
        while moved < move_limit and receiver_index < len(receivers) and donor_index < len(donors):
            receiver = receivers[receiver_index]
            donor = donors[donor_index]
            if scores[receiver] <= scores[donor] + self.config.capacity_hysteresis:
                break
            amount = min(
                desired[receiver] - result[receiver],
                result[donor] - desired[donor],
                move_limit - moved,
            )
            result[receiver] += amount
            result[donor] -= amount
            moved += amount
            if result[receiver] == desired[receiver]:
                receiver_index += 1
            if result[donor] == desired[donor]:
                donor_index += 1
        if sum(result.values()) != self.total_slots:
            raise AssertionError("capacity recommendation changed the total slot budget")
        return result, moved

    def recommended_capacities(self) -> dict[int, int]:
        return {layer_id: layer.recommended_capacity for layer_id, layer in self._layers.items()}

    def mark_applied(self, capacities: dict[int, int]) -> None:
        if set(capacities) != set(self._layers):
            raise ValueError("applied layer capacities do not cover every layer")
        if sum(capacities.values()) != self.total_slots:
            raise ValueError("applied layer capacities changed the slot budget")
        for layer_id, capacity in capacities.items():
            self._layers[layer_id].applied_capacity = int(capacity)

    def layer_metrics(self) -> dict[int, dict[str, int | float]]:
        return {
            layer_id: {
                "layer_capacity_current": layer.applied_capacity,
                "layer_capacity_recommended": layer.recommended_capacity,
                "layer_activation_ratio_ema": layer.activation_ratio_ema,
                "layer_cache_utilization_ema": layer.cache_utilization_ema,
                "layer_capacity_score": layer.allocation_score,
            }
            for layer_id, layer in self._layers.items()
        }

    def summary(self) -> dict[str, Any]:
        recommendations = {
            layer_id: layer.recommended_capacity for layer_id, layer in self._layers.items()
        }
        return {
            "enabled": self.config.enabled,
            "dry_run": self.config.enabled and not self.config.apply,
            "applied": self.config.apply,
            "completed_forwards": self.completed_forwards,
            "updates": self.update_count,
            "total_slots": self.total_slots,
            "last_moved_slots": self.last_moved_slots,
            "converged": self.converged,
            "stable_updates": self.stable_update_count,
            "current_update_interval": self.current_update_interval,
            "next_update_forward": self.next_update_forward,
            "recommended_min": min(recommendations.values(), default=0),
            "recommended_max": max(recommendations.values(), default=0),
            "layers": self.layer_metrics(),
            "history": list(self.history),
        }
