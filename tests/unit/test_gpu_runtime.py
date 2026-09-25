import threading

import pytest
import torch
import torch.nn.functional as F

from oled_moe import (
    CacheConfig,
    CPUConfig,
    OLEDConfig,
    PlacementConfig,
    ProfilingConfig,
    TransferConfig,
)
from oled_moe.runtime import MoERuntime


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_only_runtime_matches_full_resident_reference() -> None:
    torch.manual_seed(7)
    config = OLEDConfig(
        cache=CacheConfig(capacity_per_layer=4),
        transfer=TransferConfig(use_copy_stream=True),
        profiling=ProfilingConfig(enabled=True),
        pin_memory=True,
    )
    runtime = MoERuntime(config, dtype=torch.float32)
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

    hidden = torch.randn(6, 8, device="cuda", dtype=torch.float32)
    expert_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]],
        device="cuda",
    )
    weights = torch.rand(6, 2, device="cuda")
    weights /= weights.sum(dim=-1, keepdim=True)
    actual = runtime.execute(
        layer_id=1,
        hidden_states=hidden,
        expert_ids=expert_ids,
        gate_weights=weights,
    )

    expected = torch.zeros_like(hidden)
    for token_id in range(hidden.shape[0]):
        for route_id in range(expert_ids.shape[1]):
            expert_id = int(expert_ids[token_id, route_id])
            x = hidden[token_id]
            y = F.linear(
                F.silu(F.linear(x, gate[expert_id].cuda())) * F.linear(x, up[expert_id].cuda()),
                down[expert_id].cuda(),
            )
            expected[token_id] += weights[token_id, route_id] * y
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    summary = runtime.summary()
    assert summary["cache_miss_pairs"] == 12
    assert summary["h2d_expert_loads"] == 4
    assert summary["h2d_copy_submissions"] == 4
    assert summary["cuda_h2d_ms"] > 0
    assert summary["cuda_moe_compute_ms"] > 0
    assert summary["host_execute_ms"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hit_miss_overlap_and_double_buffer_matches_reference() -> None:
    torch.manual_seed(11)
    runtime = MoERuntime(
        OLEDConfig(
            cache=CacheConfig(capacity_per_layer=2),
            transfer=TransferConfig(use_copy_stream=True),
            profiling=ProfilingConfig(enabled=True),
            pin_memory=True,
        ),
        dtype=torch.float32,
    )
    runtime.register_layer(
        layer_id=1,
        num_experts=6,
        hidden_size=8,
        intermediate_size=4,
    )
    layer = runtime.store.layer(1)
    gate = torch.randn(6, 4, 8)
    up = torch.randn(6, 4, 8)
    down = torch.randn(6, 8, 4)
    for expert_id in range(6):
        layer.put(expert_id, "gate_proj", gate[expert_id])
        layer.put(expert_id, "up_proj", up[expert_id])
        layer.put(expert_id, "down_proj", down[expert_id])
    runtime.store.validate_complete()

    warm_hidden = torch.randn(2, 8, device="cuda")
    runtime.execute(
        layer_id=1,
        hidden_states=warm_hidden,
        expert_ids=torch.tensor([[0], [1]], device="cuda"),
        gate_weights=torch.ones(2, 1, device="cuda"),
    )
    torch.cuda.synchronize()
    runtime.reset_metrics()

    hidden = torch.randn(5, 8, device="cuda")
    expert_ids = torch.tensor([[0], [2], [3], [4], [5]], device="cuda")
    weights = torch.ones(5, 1, device="cuda")
    actual = runtime.execute(
        layer_id=1,
        hidden_states=hidden,
        expert_ids=expert_ids,
        gate_weights=weights,
    )

    expected = torch.zeros_like(hidden)
    for token_id, expert_id in enumerate([0, 2, 3, 4, 5]):
        x = hidden[token_id]
        expected[token_id] = F.linear(
            F.silu(F.linear(x, gate[expert_id].cuda())) * F.linear(x, up[expert_id].cuda()),
            down[expert_id].cuda(),
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    summary = runtime.summary()
    assert summary["cache_hit_pairs"] == 1
    assert summary["cache_miss_pairs"] == 4
    assert summary["h2d_expert_loads"] == 4
    assert summary["h2d_copy_submissions"] == 4
    assert summary["cuda_h2d_ms"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 3e-2, 3e-2),
    ],
)
@pytest.mark.parametrize("transfer_high_priority", [True, False])
def test_cooperative_cpu_h2d_gpu_overlap_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    transfer_high_priority: bool,
    rtol: float,
    atol: float,
) -> None:
    torch.manual_seed(29)
    runtime = MoERuntime(
        OLEDConfig(
            cache=CacheConfig(capacity_per_layer=2),
            transfer=TransferConfig(use_copy_stream=True, overlap_compute=True),
            placement=PlacementConfig(
                mode="cooperative",
                cpu_miss_ratio=0.5,
                transfer_high_priority=transfer_high_priority,
            ),
            cpu=CPUConfig(num_threads=2, strategy="expert_parallel"),
            profiling=ProfilingConfig(enabled=True),
            pin_memory=True,
        ),
        dtype=dtype,
    )
    runtime.register_layer(
        layer_id=1,
        num_experts=6,
        hidden_size=8,
        intermediate_size=4,
    )
    layer = runtime.store.layer(1)
    gate = torch.randn(6, 4, 8, dtype=dtype)
    up = torch.randn(6, 4, 8, dtype=dtype)
    down = torch.randn(6, 8, 4, dtype=dtype)
    for expert_id in range(6):
        layer.put(expert_id, "gate_proj", gate[expert_id])
        layer.put(expert_id, "up_proj", up[expert_id])
        layer.put(expert_id, "down_proj", down[expert_id])
    runtime.store.validate_complete()

    for expert_id in (0, 1):
        runtime.execute(
            layer_id=1,
            hidden_states=torch.randn(1, 8, device="cuda", dtype=dtype),
            expert_ids=torch.tensor([[expert_id]], device="cuda"),
            gate_weights=torch.ones(1, 1, device="cuda"),
        )
    torch.cuda.synchronize()
    runtime.reset_metrics()

    cpu_started = threading.Event()
    gpu_dispatched = threading.Event()
    if runtime.cpu_executor is None:
        raise AssertionError("cooperative runtime did not create the CPU executor")
    original_cpu = runtime.cpu_executor.execute_prepared
    original_gpu = runtime._execute_chunk

    def guarded_cpu(prepared):
        cpu_started.set()
        if not gpu_dispatched.wait(timeout=5):
            raise AssertionError("GPU work was not dispatched while CPU work was active")
        return original_cpu(prepared)

    def observed_gpu(**kwargs):
        if not cpu_started.wait(timeout=5):
            raise AssertionError("CPU work was not submitted before GPU execution")
        gpu_dispatched.set()
        return original_gpu(**kwargs)

    monkeypatch.setattr(runtime.cpu_executor, "execute_prepared", guarded_cpu)
    monkeypatch.setattr(runtime, "_execute_chunk", observed_gpu)

    hidden = torch.randn(5, 8, device="cuda", dtype=dtype)
    expert_ids = torch.tensor([[0], [2], [3], [4], [5]], device="cuda")
    weights = torch.ones(5, 1, device="cuda")
    actual = runtime.execute(
        layer_id=1,
        hidden_states=hidden,
        expert_ids=expert_ids,
        gate_weights=weights,
    )

    expected = torch.zeros_like(hidden)
    for token_id, expert_id in enumerate([0, 2, 3, 4, 5]):
        x = hidden[token_id]
        expected[token_id] = F.linear(
            F.silu(F.linear(x, gate[expert_id].cuda())) * F.linear(x, up[expert_id].cuda()),
            down[expert_id].cuda(),
        )
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    summary = runtime.summary()
    assert summary["cache_hit_pairs"] == 1
    assert summary["cache_miss_pairs"] == 4
    assert summary["active_experts"] == 5
    assert summary["cache_hit_experts"] == 1
    assert summary["expert_cache_hit_rate"] == pytest.approx(0.2)
    assert summary["cache_utilization"] == pytest.approx(0.5)
    assert summary["gpu_routed_pairs"] == 3
    assert summary["cpu_routed_pairs"] == 2
    assert summary["h2d_expert_loads"] == 2
    assert summary["cpu_dispatch_submissions"] == 1
    assert summary["transfer_high_priority"] is transfer_high_priority
    assert summary["host_cpu_wait_ms"] >= 0
    runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_priority_retention_keeps_top_confidence_gate_experts() -> None:
    torch.manual_seed(37)
    runtime = MoERuntime(
        OLEDConfig(
            cache=CacheConfig(capacity_per_layer=2, policy="priority"),
            transfer=TransferConfig(use_copy_stream=True, overlap_compute=True),
            profiling=ProfilingConfig(enabled=True),
            pin_memory=True,
        ),
        dtype=torch.float32,
    )
    runtime.register_layer(
        layer_id=0,
        num_experts=4,
        hidden_size=8,
        intermediate_size=4,
    )
    layer = runtime.store.layer(0)
    for expert_id in range(4):
        layer.put(expert_id, "gate_proj", torch.randn(4, 8))
        layer.put(expert_id, "up_proj", torch.randn(4, 8))
        layer.put(expert_id, "down_proj", torch.randn(8, 4))
    runtime.store.validate_complete()
    runtime.set_previous_confidence(torch.tensor([[0.9, 0.2]], device="cuda"))

    runtime.execute(
        layer_id=0,
        hidden_states=torch.randn(2, 8, device="cuda"),
        expert_ids=torch.tensor([[0, 1], [2, 3]], device="cuda"),
        gate_weights=torch.tensor([[0.8, 0.2], [0.5, 0.5]], device="cuda"),
    )
    first_summary = runtime.summary()
    assert first_summary["h2d_expert_loads"] == 4
    assert first_summary["retention_capture_device_copies"] == 0
    assert first_summary["retention_device_copies"] == 0
    assert runtime._cache(0, torch.device("cuda:0")).resident_experts == {0, 1}

    runtime.reset_metrics()
    runtime.execute(
        layer_id=0,
        hidden_states=torch.randn(2, 8, device="cuda"),
        expert_ids=torch.tensor([[0], [1]], device="cuda"),
        gate_weights=torch.ones(2, 1, device="cuda"),
    )
    second_summary = runtime.summary()
    assert second_summary["expert_cache_hit_rate"] == 1.0
    assert second_summary["cache_utilization"] == 1.0
    runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_lfu_retention_uses_cumulative_token_route_frequency() -> None:
    torch.manual_seed(41)
    runtime = MoERuntime(
        OLEDConfig(
            cache=CacheConfig(capacity_per_layer=2, policy="lfu"),
            transfer=TransferConfig(use_copy_stream=True, overlap_compute=True),
            pin_memory=True,
        ),
        dtype=torch.float32,
    )
    runtime.register_layer(
        layer_id=0,
        num_experts=4,
        hidden_size=8,
        intermediate_size=4,
    )
    layer = runtime.store.layer(0)
    for expert_id in range(4):
        layer.put(expert_id, "gate_proj", torch.randn(4, 8))
        layer.put(expert_id, "up_proj", torch.randn(4, 8))
        layer.put(expert_id, "down_proj", torch.randn(8, 4))
    runtime.store.validate_complete()

    runtime.execute(
        layer_id=0,
        hidden_states=torch.randn(5, 8, device="cuda"),
        expert_ids=torch.tensor([[0], [0], [0], [1], [2]], device="cuda"),
        gate_weights=torch.ones(5, 1, device="cuda"),
    )
    assert runtime._cache(0, torch.device("cuda:0")).resident_experts == {0, 2}

    runtime.execute(
        layer_id=0,
        hidden_states=torch.randn(5, 8, device="cuda"),
        expert_ids=torch.tensor([[3], [3], [3], [3], [0]], device="cuda"),
        gate_weights=torch.ones(5, 1, device="cuda"),
    )
    assert runtime._cache(0, torch.device("cuda:0")).resident_experts == {0, 3}
    summary = runtime.summary()
    assert summary["route_accounting_calls"] == 2
    assert summary["route_accounted_pairs"] == 10
    assert summary["retention_updates"] == 2
    assert summary["cpu_routed_pairs"] == 0
    runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_observed_lru_accounts_for_every_route_without_cpu_execution() -> None:
    runtime = MoERuntime(
        OLEDConfig(cache=CacheConfig(capacity_per_layer=2, policy="lru_observed")),
        dtype=torch.float32,
    )
    runtime.register_layer(
        layer_id=0,
        num_experts=2,
        hidden_size=4,
        intermediate_size=2,
    )
    layer = runtime.store.layer(0)
    for expert_id in range(2):
        layer.put(expert_id, "gate_proj", torch.randn(2, 4))
        layer.put(expert_id, "up_proj", torch.randn(2, 4))
        layer.put(expert_id, "down_proj", torch.randn(4, 2))
    runtime.store.validate_complete()
    runtime.execute(
        layer_id=0,
        hidden_states=torch.randn(3, 4, device="cuda"),
        expert_ids=torch.tensor([[0, 1], [0, 1], [0, 1]], device="cuda"),
        gate_weights=torch.full((3, 2), 0.5, device="cuda"),
    )
    summary = runtime.summary()
    assert summary["route_accounting_calls"] == 1
    assert summary["route_accounted_pairs"] == 6
    assert summary["retention_updates"] == 1
    assert summary["cpu_routed_pairs"] == 0
    runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_lru_explicitly_retains_the_most_recent_block_experts() -> None:
    runtime = MoERuntime(
        OLEDConfig(cache=CacheConfig(capacity_per_layer=2, policy="lru")),
        dtype=torch.float32,
    )
    runtime.register_layer(
        layer_id=0,
        num_experts=3,
        hidden_size=4,
        intermediate_size=2,
    )
    layer = runtime.store.layer(0)
    for expert_id in range(3):
        layer.put(expert_id, "gate_proj", torch.randn(2, 4))
        layer.put(expert_id, "up_proj", torch.randn(2, 4))
        layer.put(expert_id, "down_proj", torch.randn(4, 2))
    runtime.store.validate_complete()

    runtime.execute(
        layer_id=0,
        hidden_states=torch.randn(3, 4, device="cuda"),
        expert_ids=torch.tensor([[0], [1], [2]], device="cuda"),
        gate_weights=torch.ones(3, 1, device="cuda"),
    )

    summary = runtime.summary()
    assert runtime._cache(0, torch.device("cuda:0")).resident_experts == {1, 2}
    assert summary["route_accounting_calls"] == 1
    assert summary["retention_updates"] == 1
    assert summary["retention_capture_device_copies"] == 0
    assert summary["retention_device_copies"] == 0
    runtime.close()


@pytest.mark.parametrize(
    ("capacity", "expected"),
    ((20, 8), (40, 16), (100, 16), (160, 24)),
)
def test_automatic_priority_interval_tracks_cache_ratio(
    capacity: int,
    expected: int,
) -> None:
    runtime = MoERuntime(
        OLEDConfig(
            cache=CacheConfig(
                capacity_per_layer=capacity,
                policy="priority",
                update_interval=0,
            )
        ),
        dtype=torch.float32,
    )
    assert runtime._priority_update_interval(256) == expected
    runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_priority_uses_transient_slots_without_moving_retained_residents() -> None:
    torch.manual_seed(47)
    runtime = MoERuntime(
        OLEDConfig(
            cache=CacheConfig(
                capacity_per_layer=2,
                policy="priority",
            ),
            transfer=TransferConfig(use_copy_stream=True, overlap_compute=True),
            pin_memory=True,
        ),
        dtype=torch.float32,
    )
    runtime.register_layer(
        layer_id=0,
        num_experts=4,
        hidden_size=8,
        intermediate_size=4,
    )
    layer = runtime.store.layer(0)
    gate = torch.randn(4, 4, 8)
    up = torch.randn(4, 4, 8)
    down = torch.randn(4, 8, 4)
    for expert_id in range(4):
        layer.put(expert_id, "gate_proj", gate[expert_id])
        layer.put(expert_id, "up_proj", up[expert_id])
        layer.put(expert_id, "down_proj", down[expert_id])
    runtime.store.validate_complete()

    runtime.set_previous_confidence(torch.tensor([[1.0, 0.5]], device="cuda"))
    runtime.execute(
        layer_id=0,
        hidden_states=torch.randn(2, 8, device="cuda"),
        expert_ids=torch.tensor([[0], [1]], device="cuda"),
        gate_weights=torch.ones(2, 1, device="cuda"),
    )
    assert runtime._cache(0, torch.device("cuda:0")).resident_experts == {0, 1}
    runtime.reset_metrics()

    hidden = torch.randn(3, 8, device="cuda")
    expert_ids = torch.tensor([[0], [2], [3]], device="cuda")
    weights = torch.ones(3, 1, device="cuda")
    runtime.set_previous_confidence(torch.tensor([[1.0, 0.8, 0.1]], device="cuda"))
    actual = runtime.execute(
        layer_id=0,
        hidden_states=hidden,
        expert_ids=expert_ids,
        gate_weights=weights,
    )

    expected = torch.zeros_like(hidden)
    for token_id, expert_id in enumerate([0, 2, 3]):
        x = hidden[token_id]
        expected[token_id] = F.linear(
            F.silu(F.linear(x, gate[expert_id].cuda())) * F.linear(x, up[expert_id].cuda()),
            down[expert_id].cuda(),
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    summary = runtime.summary()
    assert runtime._cache(0, torch.device("cuda:0")).resident_experts == {0, 2}
    assert summary["h2d_expert_loads"] == 2
    assert summary["transient_expert_loads"] == 1
    assert summary["retention_capture_device_copies"] == 0
    assert summary["retention_device_copies"] == 0
    assert summary["gpu_eviction_buffer_bytes"] == 0
    assert summary["gpu_transient_buffer_bytes"] > 0
    runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cross_block_forward_does_not_advance_layer_capacity_controller() -> None:
    from oled_moe import LayerCapacityConfig

    runtime = MoERuntime(
        OLEDConfig(
            cache=CacheConfig(capacity_per_layer=2, policy="priority"),
            layer_capacity=LayerCapacityConfig(
                enabled=True,
                apply=True,
                warmup_forwards=1,
                update_interval=1,
                steady_update_interval=30,
                convergence_patience=3,
                ema_decay=0.0,
                min_capacity_ratio=0.5,
                max_capacity_ratio=2.0,
                max_slots_moved_ratio=0.5,
                capacity_hysteresis=0.0,
            ),
        ),
        dtype=torch.float32,
    )
    for layer_id in (0, 1):
        runtime.register_layer(
            layer_id=layer_id,
            num_experts=4,
            hidden_size=4,
            intermediate_size=2,
        )
        layer = runtime.store.layer(layer_id)
        for expert_id in range(4):
            layer.put(expert_id, "gate_proj", torch.randn(2, 4))
            layer.put(expert_id, "up_proj", torch.randn(2, 4))
            layer.put(expert_id, "down_proj", torch.randn(4, 2))
    runtime.store.validate_complete()

    runtime.begin_request(block_length=2)
    runtime.begin_model_forward(token_span=4)
    for layer_id in (0, 1):
        runtime.execute(
            layer_id=layer_id,
            hidden_states=torch.randn(4, 4, device="cuda"),
            expert_ids=torch.tensor([[0], [1], [2], [3]], device="cuda"),
            gate_weights=torch.ones(4, 1, device="cuda"),
        )
    assert runtime.capacity_plan()["completed_forwards"] == 0
    assert runtime.capacity_plan()["updates"] == 0

    runtime.begin_model_forward(token_span=2)
    for layer_id in (0, 1):
        runtime.execute(
            layer_id=layer_id,
            hidden_states=torch.randn(2, 4, device="cuda"),
            expert_ids=torch.tensor([[0], [1]], device="cuda"),
            gate_weights=torch.ones(2, 1, device="cuda"),
        )
    summary = runtime.summary()
    assert summary["layer_capacity_completed_forwards"] == 1
    assert summary["layer_capacity_updates"] == 1
    assert summary["layer_capacity_skipped_forwards"] == 1
    runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_online_layer_capacity_applies_recommendation_without_reallocation() -> None:
    from oled_moe import LayerCapacityConfig

    runtime = MoERuntime(
        OLEDConfig(
            cache=CacheConfig(capacity_per_layer=2, policy="priority"),
            layer_capacity=LayerCapacityConfig(
                enabled=True,
                apply=True,
                warmup_forwards=1,
                update_interval=1,
                ema_decay=0.0,
                min_capacity_ratio=0.5,
                max_capacity_ratio=2.0,
                max_slots_moved_ratio=0.5,
                capacity_hysteresis=0.0,
            ),
        ),
        dtype=torch.float32,
    )
    for layer_id in (0, 1):
        runtime.register_layer(
            layer_id=layer_id,
            num_experts=4,
            hidden_size=4,
            intermediate_size=2,
        )
        layer = runtime.store.layer(layer_id)
        for expert_id in range(4):
            layer.put(expert_id, "gate_proj", torch.randn(2, 4))
            layer.put(expert_id, "up_proj", torch.randn(2, 4))
            layer.put(expert_id, "down_proj", torch.randn(4, 2))
    runtime.store.validate_complete()

    runtime.execute(
        layer_id=0,
        hidden_states=torch.randn(3, 4, device="cuda"),
        expert_ids=torch.tensor([[0], [1], [2]], device="cuda"),
        gate_weights=torch.ones(3, 1, device="cuda"),
    )
    runtime.execute(
        layer_id=1,
        hidden_states=torch.randn(1, 4, device="cuda"),
        expert_ids=torch.tensor([[0]], device="cuda"),
        gate_weights=torch.ones(1, 1, device="cuda"),
    )
    assert runtime.capacity_plan()["updates"] == 1
    assert runtime.capacity_plan()["recommended_max"] == 3
    assert runtime.capacity_plan()["recommended_min"] == 1

    runtime.execute(
        layer_id=0,
        hidden_states=torch.randn(1, 4, device="cuda"),
        expert_ids=torch.tensor([[0]], device="cuda"),
        gate_weights=torch.ones(1, 1, device="cuda"),
    )
    first = runtime._cache(0, torch.device("cuda:0"))
    second = runtime._cache(1, torch.device("cuda:0"))
    assert first.capacity == 3
    assert second.capacity == 1
    assert first.pool is second.pool
    assert first.pool is not None
    assert first.pool.total_slots == 4
    summary = runtime.summary()
    assert summary["layer_capacity_apply"] is True
    assert summary["layer_capacity_applied_updates"] == 1
    assert summary["layer_capacity_total_slots"] == 4
    assert sum(cache.capacity for cache in runtime._caches.values()) == 4
    runtime.close()
