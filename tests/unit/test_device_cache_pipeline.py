import pytest
import torch

from oled_moe.memory import LayerDeviceCache, PinnedExpertStore


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_transfer_batch_does_not_evict_its_own_assignments() -> None:
    store = PinnedExpertStore(dtype=torch.float32, pin_memory=True)
    layer = store.register_layer(
        layer_id=1,
        num_experts=6,
        hidden_size=4,
        intermediate_size=2,
    )
    for expert_id in range(6):
        layer.put(expert_id, "gate_proj", torch.full((2, 4), expert_id + 1.0))
        layer.put(expert_id, "up_proj", torch.full((2, 4), expert_id + 2.0))
        layer.put(expert_id, "down_proj", torch.full((4, 2), expert_id + 3.0))

    cache = LayerDeviceCache(
        layer_id=1,
        store=layer,
        capacity=4,
        device=torch.device("cuda"),
        dtype=torch.float32,
        use_copy_stream=True,
    )
    cold = cache.begin_transfer([0, 1, 2, 3])
    assert cold.copy_submissions == 4
    cache.wait_for_transfer(cold)
    torch.cuda.synchronize()

    replacement = cache.begin_transfer([4, 5], protected_experts=[0])
    assert replacement.copy_submissions == 2
    cache.wait_for_transfer(replacement)
    torch.cuda.synchronize()

    assert {0, 4, 5}.issubset(cache.resident_experts)
    assert len({cache.slot(0), cache.slot(4), cache.slot(5)}) == 3
    torch.testing.assert_close(cache.w13[cache.slot(4), 0, 0].cpu(), torch.tensor(5.0))
    torch.testing.assert_close(cache.w13[cache.slot(5), 0, 0].cpu(), torch.tensor(6.0))
    torch.testing.assert_close(cache.w2[cache.slot(4), 0, 0].cpu(), torch.tensor(7.0))
    assert cache.w13.untyped_storage().data_ptr() == cache.packed.untyped_storage().data_ptr()
    assert cache.w2.untyped_storage().data_ptr() == cache.packed.untyped_storage().data_ptr()

    split_cache = LayerDeviceCache(
        layer_id=1,
        store=layer,
        capacity=1,
        device=torch.device("cuda"),
        dtype=torch.float32,
        use_copy_stream=True,
        coalesce_expert_weights=False,
    )
    split = split_cache.begin_transfer([0])
    split_cache.wait_for_transfer(split)
    torch.cuda.synchronize()
    assert split.copy_submissions == 2
    torch.testing.assert_close(
        split_cache.w2[split_cache.slot(0), 0, 0].cpu(),
        torch.tensor(3.0),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_runtime_layer_caches_share_one_global_device_pool() -> None:
    from oled_moe import CacheConfig, OLEDConfig
    from oled_moe.runtime import MoERuntime

    runtime = MoERuntime(
        OLEDConfig(cache=CacheConfig(capacity_per_layer=2)),
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

    for layer_id in (0, 1):
        runtime.execute(
            layer_id=layer_id,
            hidden_states=torch.randn(2, 4, device="cuda"),
            expert_ids=torch.tensor([[0], [1]], device="cuda"),
            gate_weights=torch.ones(2, 1, device="cuda"),
        )
    torch.cuda.synchronize()

    first = runtime._cache(0, torch.device("cuda:0"))
    second = runtime._cache(1, torch.device("cuda:0"))
    assert first.pool is second.pool
    assert first.pool is not None
    assert first.pool.total_slots == 4
    assert first.pool.free_slots == 0
    assert set(first.expert_to_slot.values()).isdisjoint(second.expert_to_slot.values())
    assert runtime.summary()["gpu_cache_pool_count"] == 1
    runtime.close()
