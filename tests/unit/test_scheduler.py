import pytest

from oled_moe.core import DispatchPlan, ExpertKey
from oled_moe.scheduler import CooperativeMissScheduler, GPUOnlyMissScheduler


def test_gpu_only_scheduler_preserves_priority_and_never_emits_cpu_work() -> None:
    plan = GPUOnlyMissScheduler().plan(
        hits=[ExpertKey(1, 2)],
        misses=[ExpertKey(1, 4), ExpertKey(1, 3)],
    )
    assert plan.hit_experts == (ExpertKey(1, 2),)
    assert plan.transfer_experts == (ExpertKey(1, 4), ExpertKey(1, 3))
    assert plan.cpu_experts == ()


def test_cooperative_scheduler_transfers_high_priority_misses() -> None:
    misses = [ExpertKey(1, expert) for expert in (9, 7, 5, 3)]
    priorities = {
        misses[0]: 0.2,
        misses[1]: 0.9,
        misses[2]: 0.1,
        misses[3]: 0.5,
    }
    plan = CooperativeMissScheduler(
        cpu_miss_ratio=0.5,
        transfer_high_priority=True,
    ).plan(
        hits=[ExpertKey(1, 11)],
        misses=misses,
        priorities=priorities,
    )
    assert plan.hit_experts == (ExpertKey(1, 11),)
    assert plan.transfer_experts == (misses[1], misses[3])
    assert plan.cpu_experts == (misses[0], misses[2])


def test_cooperative_scheduler_can_transfer_low_priority_misses() -> None:
    misses = [ExpertKey(1, expert) for expert in (9, 7, 5, 3)]
    priorities = {
        misses[0]: 0.2,
        misses[1]: 0.9,
        misses[2]: 0.1,
        misses[3]: 0.5,
    }
    plan = CooperativeMissScheduler(
        cpu_miss_ratio=0.5,
        transfer_high_priority=False,
    ).plan(hits=(), misses=misses, priorities=priorities)
    assert plan.transfer_experts == (misses[2], misses[0])
    assert plan.cpu_experts == (misses[3], misses[1])


def test_cooperative_scheduler_uses_expert_id_as_stable_tie_break() -> None:
    misses = [ExpertKey(1, expert) for expert in (9, 3, 7, 5)]
    priorities = dict.fromkeys(misses, 0.5)
    plan = CooperativeMissScheduler(cpu_miss_ratio=0.5).plan(
        hits=(),
        misses=misses,
        priorities=priorities,
    )
    assert plan.transfer_experts == (ExpertKey(1, 3), ExpertKey(1, 5))
    assert plan.cpu_experts == (ExpertKey(1, 7), ExpertKey(1, 9))



def test_cooperative_scheduler_accumulates_small_ratios_across_layers() -> None:
    scheduler = CooperativeMissScheduler(cpu_miss_ratio=0.02)
    misses = [ExpertKey(1, expert) for expert in range(10)]

    plans = [
        scheduler.plan(hits=(), misses=misses, priorities=None)
        for _ in range(10)
    ]

    assert [len(plan.cpu_experts) for plan in plans] == [0, 0, 0, 0, 1] * 2
    assert sum(len(plan.cpu_experts) for plan in plans) == 2


def test_cooperative_scheduler_does_not_force_cpu_work_per_layer() -> None:
    scheduler = CooperativeMissScheduler(cpu_miss_ratio=0.05)
    misses = [ExpertKey(1, expert) for expert in range(8)]

    plan = scheduler.plan(hits=(), misses=misses, priorities=None)

    assert len(plan.transfer_experts) == 8
    assert plan.cpu_experts == ()


def test_cooperative_scheduler_can_coarsen_cpu_dispatches() -> None:
    scheduler = CooperativeMissScheduler(
        cpu_miss_ratio=0.1,
        cpu_min_dispatch_experts=4,
    )
    misses = [ExpertKey(1, expert) for expert in range(10)]

    plans = [
        scheduler.plan(hits=(), misses=misses, priorities=None)
        for _ in range(8)
    ]

    assert [len(plan.cpu_experts) for plan in plans] == [0, 0, 0, 4] * 2
    assert sum(len(plan.cpu_experts) for plan in plans) == 8


def test_cooperative_scheduler_rejects_invalid_minimum_dispatch() -> None:
    with pytest.raises(ValueError, match="cpu_min_dispatch_experts"):
        CooperativeMissScheduler(
            cpu_miss_ratio=0.1,
            cpu_min_dispatch_experts=0,
        )

def test_dispatch_plan_rejects_overlapping_partitions() -> None:
    expert = ExpertKey(1, 2)
    with pytest.raises(ValueError, match="partitions must be disjoint"):
        DispatchPlan(hit_experts=(expert,), transfer_experts=(expert,))
