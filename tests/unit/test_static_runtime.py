import pytest
import torch
import torch.nn.functional as F

from oled_moe import (
    CacheConfig,
    CPUConfig,
    OLEDConfig,
    PlacementConfig,
    ProfilingConfig,
)
from oled_moe.runtime import MoERuntime


def _populate_runtime(runtime: MoERuntime) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    runtime.register_layer(
        layer_id=1,
        num_experts=4,
        hidden_size=8,
        intermediate_size=4,
    )
    layer = runtime.store.layer(1)
    gate = torch.randn(4, 4, 8)
    up = torch.randn(4, 4, 8)
    down = torch.randn(4, 8, 4)
    for expert_id in range(4):
        layer.put(expert_id, "gate_proj", gate[expert_id])
        layer.put(expert_id, "up_proj", up[expert_id])
        layer.put(expert_id, "down_proj", down[expert_id])
    runtime.store.validate_complete()
    return gate, up, down


def _reference(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    expected = torch.zeros_like(hidden)
    for token_id in range(hidden.shape[0]):
        for route_id in range(expert_ids.shape[1]):
            expert_id = int(expert_ids[token_id, route_id])
            y = F.linear(
                F.silu(F.linear(hidden[token_id], gate[expert_id].cuda()))
                * F.linear(hidden[token_id], up[expert_id].cuda()),
                down[expert_id].cuda(),
            )
            expected[token_id] += weights[token_id, route_id] * y
    return expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("placement", "expected_gpu_pairs", "expected_cpu_pairs", "expected_gpu_bytes"),
    [
        (PlacementConfig(mode="cpu_only"), 0, 8, 0),
        (
            PlacementConfig(mode="static_hybrid", static_gpu_experts_per_layer=2),
            4,
            4,
            2 * 3 * 4 * 8 * torch.tensor([], dtype=torch.float32).element_size(),
        ),
    ],
)
def test_static_runtime_matches_reference_without_runtime_weight_loads(
    placement: PlacementConfig,
    expected_gpu_pairs: int,
    expected_cpu_pairs: int,
    expected_gpu_bytes: int,
) -> None:
    torch.manual_seed(23)
    runtime = MoERuntime(
        OLEDConfig(
            cache=CacheConfig(capacity_per_layer=2),
            placement=placement,
            cpu=CPUConfig(num_threads=2),
            profiling=ProfilingConfig(enabled=True),
            pin_memory=True,
        ),
        dtype=torch.float32,
    )
    gate, up, down = _populate_runtime(runtime)
    runtime.prepare(torch.device("cuda"))
    runtime.reset_metrics()

    hidden = torch.randn(4, 8, device="cuda")
    expert_ids = torch.tensor([[0, 2], [1, 3], [0, 3], [1, 2]], device="cuda")
    weights = torch.rand(4, 2, device="cuda")
    weights /= weights.sum(dim=-1, keepdim=True)
    actual = runtime.execute(
        layer_id=1,
        hidden_states=hidden,
        expert_ids=expert_ids,
        gate_weights=weights,
    )
    expected = _reference(hidden, expert_ids, weights, gate, up, down)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    summary = runtime.summary()
    assert summary["gpu_routed_pairs"] == expected_gpu_pairs
    assert summary["cpu_routed_pairs"] == expected_cpu_pairs
    assert summary["h2d_expert_loads"] == 0
    assert summary["gpu_cache_bytes"] == expected_gpu_bytes
    assert summary["host_cpu_compute_ms"] > 0
    runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_static_gpu_runtime_loads_misses_transiently_without_changing_residents() -> None:
    torch.manual_seed(29)
    runtime = MoERuntime(
        OLEDConfig(
            cache=CacheConfig(capacity_per_layer=2),
            placement=PlacementConfig(
                mode="static_gpu",
                static_gpu_experts_per_layer=2,
            ),
            profiling=ProfilingConfig(enabled=True),
            pin_memory=True,
        ),
        dtype=torch.float32,
    )
    gate, up, down = _populate_runtime(runtime)
    runtime.prepare(torch.device("cuda"))
    runtime.reset_metrics()

    hidden = torch.randn(4, 8, device="cuda")
    expert_ids = torch.tensor([[0, 2], [1, 3], [0, 3], [1, 2]], device="cuda")
    weights = torch.rand(4, 2, device="cuda")
    weights /= weights.sum(dim=-1, keepdim=True)
    actual = runtime.execute(
        layer_id=1,
        hidden_states=hidden,
        expert_ids=expert_ids,
        gate_weights=weights,
    )
    expected = _reference(hidden, expert_ids, weights, gate, up, down)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert runtime._caches[1].resident_experts == {0, 1}
    summary = runtime.summary()
    assert summary["gpu_routed_pairs"] == 8
    assert summary["cpu_routed_pairs"] == 0
    assert summary["h2d_expert_loads"] == 2
    assert summary["transient_expert_loads"] == 2
    assert summary["cache_utilization"] == 1.0
    runtime.close()
