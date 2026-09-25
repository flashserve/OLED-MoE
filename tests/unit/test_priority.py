import numpy as np
import torch

from oled_moe import CacheConfig, OLEDConfig
from oled_moe.placement import (
    aggregate_priority_route_packet,
    align_previous_confidence,
    build_priority_route_packet,
    compute_expert_priorities,
    observe_and_rank_retention_candidates,
    select_retained_experts,
)
from oled_moe.runtime import MoERuntime


def test_priority_matches_confidence_gate_formula() -> None:
    expert_ids = torch.tensor([[0, 1], [1, 2]])
    gate_weights = torch.tensor([[0.75, 0.25], [0.4, 0.6]])
    confidence = torch.tensor([[0.8, 0.5]])

    priorities, counts = compute_expert_priorities(
        expert_ids=expert_ids,
        gate_weights=gate_weights,
        previous_confidence=confidence,
        num_experts=4,
    )

    expected = torch.tensor([0.6, 0.4, 0.3, 0.0])
    torch.testing.assert_close(priorities, expected)
    torch.testing.assert_close(counts, torch.tensor([1, 2, 1, 0]))


def test_host_packet_matches_device_priority_formula() -> None:
    expert_ids = torch.tensor([[0, 1], [1, 2]])
    gate_weights = torch.tensor([[0.75, 0.25], [0.4, 0.6]])
    confidence = torch.tensor([[0.8, 0.5]])

    packet = build_priority_route_packet(
        expert_ids=expert_ids,
        gate_weights=gate_weights,
        previous_confidence=confidence,
    )
    rows = aggregate_priority_route_packet(packet.numpy(), num_experts=4)

    expected = np.array([0.6, 0.4, 0.3])
    assert [row[:2] for row in rows] == [(0, 1), (1, 2), (2, 1)]
    np.testing.assert_allclose([row[2] for row in rows], expected, rtol=1e-6)


def test_priority_packet_optionally_preserves_unpowered_gate_weights() -> None:
    expert_ids = torch.tensor([[0, 1]])
    gate_weights = torch.tensor([[0.75, 0.25]])
    packet = build_priority_route_packet(
        expert_ids=expert_ids,
        gate_weights=gate_weights,
        previous_confidence=torch.tensor([0.8]),
        gate_power=2.0,
        include_raw_gates=True,
    )

    assert packet.shape == (2, 3)
    torch.testing.assert_close(packet[:, 1], torch.tensor([0.45, 0.05]))
    torch.testing.assert_close(packet[:, 2], torch.tensor([0.75, 0.25]))


def test_common_observation_updates_all_signals_before_policy_ranking() -> None:
    packet = np.array(
        [[0.0, 0.1], [1.0, 0.4], [0.0, 0.1], [2.0, 0.3]],
        dtype=np.float64,
    )
    expected_rankings = {
        "lru": (2, 0, 1, 3),
        "lfu": (3, 1, 2, 0),
        "priority": (1, 2, 0, 3),
    }
    observed_states = []

    for policy, expected_ranking in expected_rankings.items():
        frequency = np.array([0, 5, 1, 10], dtype=np.int64)
        gate_mass = np.zeros(4, dtype=np.float64)
        ranking = observe_and_rank_retention_candidates(
            route_packet=packet,
            token_count=2,
            num_experts=4,
            resident_experts=(3,),
            previous_recency={3: 10, 0: 2},
            route_frequency=frequency,
            route_gate_mass=gate_mass,
            policy=policy,  # type: ignore[arg-type]
        )

        assert ranking.active_experts == (0, 1, 2)
        assert ranking.route_counts == {0: 2, 1: 1, 2: 1}
        assert ranking.recency == {3: 10, 0: 13, 1: 12, 2: 14}
        assert ranking.ranked_experts == expected_ranking
        assert set(ranking.lru_scores) == {0, 1, 2, 3}
        assert set(ranking.lfu_scores) == {0, 1, 2, 3}
        assert set(ranking.priority_scores) == {0, 1, 2, 3}
        observed_states.append((frequency.copy(), gate_mass.copy()))

    for frequency, gate_mass in observed_states:
        np.testing.assert_array_equal(frequency, np.array([2, 6, 2, 10]))
        np.testing.assert_allclose(gate_mass, np.array([0.2, 0.4, 0.3, 0.0]))


def test_confidence_alignment_pads_cross_block_prefix_with_ones() -> None:
    confidence = torch.tensor([[0.2, 0.4]])
    actual = align_previous_confidence(
        confidence,
        token_count=4,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(actual, torch.tensor([1.0, 1.0, 0.2, 0.4]))


def test_empty_block_confidence_falls_back_to_ones() -> None:
    actual = align_previous_confidence(
        torch.empty((0, 0)),
        token_count=4,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(actual, torch.ones(4))


def test_retention_uses_priority_then_recency() -> None:
    retained = select_retained_experts(
        [0, 1, 2, 3],
        priorities={0: 0.1, 1: 0.8, 2: 0.8, 3: 0.2},
        recency={1: 3, 2: 7},
        capacity=2,
    )
    assert retained == (2, 1)


def test_selective_priority_targets_layers_with_largest_opportunity() -> None:
    runtime = MoERuntime(
        OLEDConfig(
            cache=CacheConfig(
                policy="priority",
                selective_priority=True,
                priority_layer_fraction=0.5,
                min_priority_opportunity=0.02,
                priority_hysteresis=0.0,
            )
        ),
        dtype=torch.float32,
    )
    runtime._layer_priority_opportunity_ema.update({0: 0.10, 1: 0.30, 2: 0.20, 3: 0.01})
    runtime._layer_priority_observations.update({layer: 1 for layer in range(4)})

    assert runtime._priority_enabled_for_layer(0) is False
    assert runtime._priority_enabled_for_layer(1) is True
    assert runtime._priority_enabled_for_layer(2) is True
    assert runtime._priority_enabled_for_layer(3) is False
    runtime.close()


def test_priority_opportunity_ignores_unavoidably_empty_slots() -> None:
    runtime = MoERuntime(
        OLEDConfig(cache=CacheConfig(policy="priority", utilization_ema_decay=0.5)),
        dtype=torch.float32,
    )

    first = runtime._observe_layer_priority_opportunity(
        layer_id=0,
        active_experts=20,
        hit_experts=20,
        capacity=100,
    )
    second = runtime._observe_layer_priority_opportunity(
        layer_id=0,
        active_experts=80,
        hit_experts=40,
        capacity=100,
    )

    assert first == 0.0
    assert second == 0.2
    runtime.close()
