import pytest

from oled_moe import CacheConfig, CPUConfig, PlacementConfig, TransferConfig


def test_cache_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        CacheConfig(capacity_per_layer=0)


def test_cache_policy_and_priority_powers_must_be_valid() -> None:
    with pytest.raises(ValueError, match="unsupported cache policy"):
        CacheConfig(policy="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confidence_power"):
        CacheConfig(policy="priority", confidence_power=0)
    with pytest.raises(ValueError, match="gate_power"):
        CacheConfig(policy="priority", gate_power=0)
    with pytest.raises(ValueError, match="update_interval"):
        CacheConfig(policy="priority", update_interval=-1)
    with pytest.raises(ValueError, match="priority_layer_fraction"):
        CacheConfig(policy="priority", priority_layer_fraction=0)
    with pytest.raises(ValueError, match="utilization_ema_decay"):
        CacheConfig(policy="priority", utilization_ema_decay=1.0)
    with pytest.raises(ValueError, match="min_priority_opportunity"):
        CacheConfig(policy="priority", min_priority_opportunity=-0.1)
    with pytest.raises(ValueError, match="priority_hysteresis"):
        CacheConfig(policy="priority", priority_hysteresis=1.1)


def test_priority_defaults_cover_every_layer_call() -> None:
    config = CacheConfig(policy="priority")
    assert config.update_interval == 1
    assert config.selective_priority is False
    assert config.priority_layer_fraction == 0.5


def test_observed_lru_and_lfu_are_supported() -> None:
    assert CacheConfig(policy="lru_observed").policy == "lru_observed"
    assert CacheConfig(policy="lfu").policy == "lfu"


def test_gpu_expert_batch_limit_rejects_singleton() -> None:
    with pytest.raises(ValueError, match="at least two"):
        TransferConfig(gpu_expert_batch_size=1)


def test_static_hybrid_requires_gpu_experts() -> None:
    with pytest.raises(ValueError):
        PlacementConfig(mode="static_hybrid")


def test_cpu_threads_must_be_positive() -> None:
    with pytest.raises(ValueError):
        CPUConfig(num_threads=0)


def test_cpu_strategy_must_be_supported() -> None:
    with pytest.raises(ValueError, match="unsupported CPU execution strategy"):
        CPUConfig(strategy="unknown")  # type: ignore[arg-type]


def test_cpu_pointwise_mode_must_be_supported() -> None:
    with pytest.raises(ValueError, match="unsupported CPU pointwise mode"):
        CPUConfig(pointwise_mode="unknown")  # type: ignore[arg-type]


def test_cpu_miss_ratio_must_be_bounded() -> None:
    with pytest.raises(ValueError, match="cpu_miss_ratio"):
        PlacementConfig(mode="cooperative", cpu_miss_ratio=1.1)


def test_cpu_min_dispatch_experts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="cpu_min_dispatch_experts"):
        PlacementConfig(mode="cooperative", cpu_min_dispatch_experts=0)


def test_static_gpu_requires_and_accepts_resident_experts() -> None:
    with pytest.raises(ValueError):
        PlacementConfig(mode="static_gpu")
    config = PlacementConfig(
        mode="static_gpu",
        static_gpu_experts_per_layer=2,
    )
    assert config.static_gpu_experts_per_layer == 2
