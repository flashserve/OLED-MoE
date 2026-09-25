import pytest

from oled_moe import LayerCapacityConfig
from oled_moe.placement import LayerCapacityController


def _register_three(controller: LayerCapacityController, capacity: int = 10) -> None:
    for layer_id in range(3):
        controller.register_layer(
            layer_id=layer_id,
            num_experts=20,
            initial_capacity=capacity,
        )


def test_dry_run_preserves_budget_and_limits_each_update() -> None:
    controller = LayerCapacityController(
        LayerCapacityConfig(
            enabled=True,
            warmup_forwards=2,
            update_interval=2,
            ema_decay=0.0,
            max_slots_moved_ratio=0.2,
            capacity_hysteresis=0.0,
        )
    )
    _register_three(controller)

    samples = ((10, 10), (5, 2), (1, 0))
    for _ in range(2):
        for layer_id, (active, hits) in enumerate(samples):
            controller.observe(
                layer_id=layer_id,
                active_experts=active,
                hit_experts=hits,
                capacity=10,
            )

    summary = controller.summary()
    recommendations = {
        layer_id: values["layer_capacity_recommended"]
        for layer_id, values in summary["layers"].items()
    }
    assert summary["completed_forwards"] == 2
    assert summary["updates"] == 1
    assert summary["total_slots"] == 30
    assert summary["last_moved_slots"] <= 6
    assert sum(recommendations.values()) == 30
    assert recommendations[0] > 10
    assert recommendations[2] < 10
    assert all(values["layer_capacity_current"] == 10 for values in summary["layers"].values())


def test_update_interval_uses_complete_transformer_forwards() -> None:
    controller = LayerCapacityController(
        LayerCapacityConfig(
            enabled=True,
            warmup_forwards=1,
            update_interval=2,
        )
    )
    _register_three(controller)

    for layer_id in (0, 1):
        controller.observe(
            layer_id=layer_id,
            active_experts=4,
            hit_experts=2,
            capacity=10,
        )
    assert controller.summary()["completed_forwards"] == 0

    controller.observe(
        layer_id=2,
        active_experts=4,
        hit_experts=2,
        capacity=10,
    )
    assert controller.summary()["completed_forwards"] == 1
    assert controller.summary()["updates"] == 1

    for _ in range(2):
        for layer_id in range(3):
            controller.observe(
                layer_id=layer_id,
                active_experts=4,
                hit_experts=2,
                capacity=10,
            )
    assert controller.summary()["completed_forwards"] == 3
    assert controller.summary()["updates"] == 2


def test_disabled_controller_does_not_collect_runtime_observations() -> None:
    controller = LayerCapacityController(LayerCapacityConfig(enabled=False))
    _register_three(controller)
    assert (
        controller.observe(
            layer_id=0,
            active_experts=10,
            hit_experts=5,
            capacity=10,
        )
        is False
    )
    assert controller.summary()["completed_forwards"] == 0
    assert controller.summary()["updates"] == 0


def test_reconfigure_resets_the_dry_run_budget() -> None:
    controller = LayerCapacityController(LayerCapacityConfig(enabled=True))
    _register_three(controller)
    controller.reset(initial_capacity=6)
    summary = controller.summary()
    assert summary["total_slots"] == 18
    assert summary["recommended_min"] == 6
    assert summary["recommended_max"] == 6


def test_controller_slows_after_convergence_and_resumes_fast_updates() -> None:
    controller = LayerCapacityController(
        LayerCapacityConfig(
            enabled=True,
            warmup_forwards=1,
            update_interval=2,
            steady_update_interval=5,
            convergence_patience=2,
            convergence_moved_ratio=0.01,
            ema_decay=0.0,
            max_slots_moved_ratio=0.2,
            capacity_hysteresis=0.0,
        )
    )
    _register_three(controller)

    uniform = ((4, 2), (4, 2), (4, 2))
    skewed = ((18, 10), (4, 2), (1, 0))

    def observe_forward(samples: tuple[tuple[int, int], ...]) -> None:
        for layer_id, (active, hits) in enumerate(samples):
            controller.observe(
                layer_id=layer_id,
                active_experts=active,
                hit_experts=hits,
                capacity=10,
            )

    observe_forward(uniform)
    assert controller.summary()["current_update_interval"] == 2
    for _ in range(2):
        observe_forward(uniform)
    summary = controller.summary()
    assert summary["updates"] == 2
    assert summary["converged"] is True
    assert summary["current_update_interval"] == 5
    assert summary["next_update_forward"] == 8

    for _ in range(4):
        observe_forward(skewed)
    assert controller.summary()["updates"] == 2
    observe_forward(skewed)
    summary = controller.summary()
    assert summary["updates"] == 3
    assert summary["converged"] is False
    assert summary["current_update_interval"] == 2
    assert summary["next_update_forward"] == 10


@pytest.mark.parametrize(
    "kwargs",
    (
        {"warmup_forwards": 0},
        {"update_interval": 0},
        {"steady_update_interval": 4},
        {"convergence_patience": 0},
        {"convergence_moved_ratio": 0.0},
        {"ema_decay": 1.0},
        {"activation_power": 0.0},
        {"utilization_power": 0.0},
        {"min_capacity_ratio": 0.0},
        {"max_capacity_ratio": 0.5},
        {"max_slots_moved_ratio": 0.0},
        {"capacity_hysteresis": -0.1},
    ),
)
def test_layer_capacity_config_rejects_invalid_values(
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        LayerCapacityConfig(**kwargs)  # type: ignore[arg-type]
