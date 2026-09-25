from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from oled_moe.config import CachePolicy


def align_previous_confidence(
    previous_confidence: torch.Tensor | None,
    *,
    token_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Align block confidence to flattened MoE tokens without a host handoff."""

    if previous_confidence is None:
        return torch.ones(token_count, device=device, dtype=torch.float32)
    confidence = previous_confidence.detach().to(device=device, dtype=torch.float32)
    if confidence.numel() == 0:
        return torch.ones(token_count, device=device, dtype=torch.float32)
    if confidence.ndim == 1:
        confidence = confidence.unsqueeze(0)
    elif confidence.ndim != 2:
        raise ValueError("previous confidence must have shape [T] or [B,T]")
    batch_size, source_length = confidence.shape
    if token_count % batch_size:
        raise ValueError(
            f"cannot align {batch_size} confidence rows to {token_count} routed tokens"
        )
    target_length = token_count // batch_size
    if source_length == target_length:
        return confidence.reshape(-1)
    if source_length > target_length:
        return confidence[:, -target_length:].reshape(-1)
    aligned = torch.ones(
        (batch_size, target_length),
        device=device,
        dtype=torch.float32,
    )
    aligned[:, -source_length:] = confidence
    return aligned.reshape(-1)


def compute_expert_priorities(
    *,
    expert_ids: torch.Tensor,
    gate_weights: torch.Tensor,
    previous_confidence: torch.Tensor | None,
    num_experts: int,
    confidence_power: float = 1.0,
    gate_power: float = 1.0,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute current-forward paper priority and route counts on device."""

    token_count = expert_ids.shape[0]
    confidence = align_previous_confidence(
        previous_confidence,
        token_count=token_count,
        device=expert_ids.device,
    )
    if confidence_power != 1.0:
        confidence = confidence.pow(confidence_power)
    gates = gate_weights.float()
    if gate_power != 1.0:
        gates = gates.pow(gate_power)
    contributions = gates * confidence.unsqueeze(-1)
    flat_experts = expert_ids.reshape(-1).long()
    if out is None:
        priorities = torch.zeros(num_experts, device=expert_ids.device, dtype=torch.float32)
    else:
        if out.shape != (num_experts,) or out.device != expert_ids.device:
            raise ValueError("priority output buffer has the wrong shape or device")
        priorities = out.zero_()
    priorities.scatter_add_(0, flat_experts, contributions.reshape(-1))
    counts = torch.bincount(flat_experts, minlength=num_experts)
    return priorities, counts


def build_priority_route_packet(
    *,
    expert_ids: torch.Tensor,
    gate_weights: torch.Tensor,
    previous_confidence: torch.Tensor | None,
    confidence_power: float = 1.0,
    gate_power: float = 1.0,
    include_raw_gates: bool = False,
) -> torch.Tensor:
    """Pack route IDs and priorities, optionally adding raw gates for CPU I/O."""

    confidence = align_previous_confidence(
        previous_confidence,
        token_count=expert_ids.shape[0],
        device=expert_ids.device,
    )
    if confidence_power != 1.0:
        confidence = confidence.pow(confidence_power)
    raw_gates = gate_weights.float()
    priority_gates = (
        raw_gates.pow(gate_power) if gate_power != 1.0 else raw_gates
    )
    contributions = priority_gates * confidence.unsqueeze(-1)
    columns = [
        expert_ids.reshape(-1).float(),
        contributions.reshape(-1),
    ]
    if include_raw_gates:
        columns.append(raw_gates.reshape(-1))
    return torch.stack(columns, dim=1)


def aggregate_priority_route_packet(
    packet: np.ndarray,
    *,
    num_experts: int,
) -> list[tuple[int, int, float]]:
    """Aggregate a small host packet without launching GPU scatter kernels."""

    if packet.ndim != 2 or packet.shape[1] not in {2, 3}:
        raise ValueError(
            "priority route packet must have shape [routes, 2] or [routes, 3]"
        )
    expert_ids = packet[:, 0].astype(np.int64, copy=False)
    if expert_ids.size and (int(expert_ids.min()) < 0 or int(expert_ids.max()) >= num_experts):
        raise ValueError("priority route packet contains an invalid expert ID")
    counts = np.bincount(expert_ids, minlength=num_experts)
    priorities = np.bincount(
        expert_ids,
        weights=packet[:, 1],
        minlength=num_experts,
    )
    active = np.flatnonzero(counts)
    return [(int(expert), int(counts[expert]), float(priorities[expert])) for expert in active]


@dataclass(frozen=True)
class RetentionRanking:
    """Common cache observations and the policy-selected expert ordering."""

    active_experts: tuple[int, ...]
    route_counts: dict[int, int]
    current_priorities: dict[int, float]
    recency: dict[int, int]
    lru_scores: dict[int, float]
    lfu_scores: dict[int, float]
    priority_scores: dict[int, float]
    selected_scores: dict[int, float]
    ranked_experts: tuple[int, ...]


def rank_retention_candidates(
    candidates: Iterable[int],
    *,
    scores: Mapping[int, float],
    recency: Mapping[int, int],
) -> tuple[int, ...]:
    """Return every candidate in deterministic highest-value-first order."""

    unique = tuple(dict.fromkeys(int(expert) for expert in candidates))
    return tuple(
        sorted(
            unique,
            key=lambda expert: (
                scores.get(expert, 0.0),
                recency.get(expert, -1),
                -expert,
            ),
            reverse=True,
        )
    )


def observe_and_rank_retention_candidates(
    *,
    route_packet: np.ndarray,
    token_count: int,
    num_experts: int,
    resident_experts: Iterable[int],
    previous_recency: Mapping[int, int],
    route_frequency: np.ndarray,
    route_gate_mass: np.ndarray,
    policy: CachePolicy,
) -> RetentionRanking:
    """Update LRU, LFU, and paper-priority state, then return one policy ordering.

    Every dynamic cache policy executes the same observation work. The selected
    policy changes only which already-computed score vector orders the candidates.
    Stateful NumPy arrays are updated in place so the runtime can reuse them.
    """

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    expected_shape = (num_experts,)
    for name, values in (
        ("route_frequency", route_frequency),
        ("route_gate_mass", route_gate_mass),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")

    rows = aggregate_priority_route_packet(route_packet, num_experts=num_experts)
    active_experts = tuple(int(expert) for expert, _, _ in rows)
    route_counts = {int(expert): int(count) for expert, count, _ in rows}
    current_priorities = {int(expert): float(priority) for expert, _, priority in rows}

    route_expert_ids = route_packet[:, 0].astype(np.int64, copy=False)
    last_route_positions = np.full(num_experts, -1, dtype=np.int64)
    np.maximum.at(
        last_route_positions,
        route_expert_ids,
        np.arange(route_expert_ids.size, dtype=np.int64),
    )
    previous_most_recent = max(previous_recency.values(), default=-1)
    recency = dict(previous_recency)
    recency.update(
        {
            expert: previous_most_recent + 1 + int(last_route_positions[expert])
            for expert in active_experts
        }
    )

    active_ids = np.fromiter(
        active_experts,
        dtype=np.int64,
        count=len(active_experts),
    )
    active_counts = np.fromiter(
        (route_counts[expert] for expert in active_experts),
        dtype=np.int64,
        count=len(active_experts),
    )
    active_gate_mass = np.fromiter(
        (current_priorities[expert] for expert in active_experts),
        dtype=np.float64,
        count=len(active_experts),
    )
    route_frequency[active_ids] += active_counts
    route_gate_mass[active_ids] += active_gate_mass

    candidates = tuple(
        dict.fromkeys(
            (
                *(int(expert) for expert in resident_experts),
                *active_experts,
            )
        )
    )
    lru_scores = {expert: float(recency.get(expert, -1)) for expert in candidates}
    lfu_scores = {expert: float(route_frequency[expert]) for expert in candidates}
    priority_scores = {expert: current_priorities.get(expert, 0.0) for expert in candidates}
    if policy in {"lru", "lru_observed"}:
        selected_scores = lru_scores
    elif policy == "lfu":
        selected_scores = lfu_scores
    elif policy == "priority":
        selected_scores = priority_scores
    else:
        raise ValueError(f"unsupported dynamic cache policy: {policy}")
    ranked_experts = rank_retention_candidates(
        candidates,
        scores=selected_scores,
        recency=recency,
    )
    return RetentionRanking(
        active_experts=active_experts,
        route_counts=route_counts,
        current_priorities=current_priorities,
        recency=recency,
        lru_scores=lru_scores,
        lfu_scores=lfu_scores,
        priority_scores=priority_scores,
        selected_scores=selected_scores,
        ranked_experts=ranked_experts,
    )


def select_retained_experts(
    candidates: Iterable[int],
    *,
    priorities: Mapping[int, float],
    recency: Mapping[int, int],
    capacity: int,
) -> tuple[int, ...]:
    """Select the exact top-C candidates with deterministic recency tie-breaking."""

    ranked = rank_retention_candidates(
        candidates,
        scores=priorities,
        recency=recency,
    )
    return ranked[:capacity]
