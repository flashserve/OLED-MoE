import numpy as np
import pytest
import torch
import torch.nn.functional as F

from oled_moe.executor import CPUExpertExecutor
from oled_moe.memory import PinnedExpertStore


@pytest.mark.parametrize("strategy", ["serial", "expert_parallel"])
@pytest.mark.parametrize("pointwise_mode", ["baseline", "inplace"])
def test_cpu_executor_matches_token_reference(
    strategy: str,
    pointwise_mode: str,
) -> None:
    torch.manual_seed(19)
    store = PinnedExpertStore(dtype=torch.float32, pin_memory=False)
    layer = store.register_layer(
        layer_id=1,
        num_experts=3,
        hidden_size=6,
        intermediate_size=4,
    )
    gate = torch.randn(3, 4, 6)
    up = torch.randn(3, 4, 6)
    down = torch.randn(3, 6, 4)
    for expert_id in range(3):
        layer.put(expert_id, "gate_proj", gate[expert_id])
        layer.put(expert_id, "up_proj", up[expert_id])
        layer.put(expert_id, "down_proj", down[expert_id])

    hidden = torch.randn(5, 6)
    expert_ids = torch.tensor([[0, 1], [1, 2], [0, 2], [2, 1], [0, 0]])
    weights = torch.rand(5, 2)
    weights /= weights.sum(dim=-1, keepdim=True)
    executor = CPUExpertExecutor(
        num_threads=2,
        strategy=strategy,
        pointwise_mode=pointwise_mode,
    )
    result = executor.execute(
        store=layer,
        hidden_states=hidden,
        expert_ids=expert_ids,
        gate_weights=weights,
        active_experts=(1, 2),
        route_counts={1: 3, 2: 3},
        profile=True,
    )

    expected = torch.zeros_like(hidden)
    for token_id in range(hidden.shape[0]):
        for route_id in range(expert_ids.shape[1]):
            expert_id = int(expert_ids[token_id, route_id])
            if expert_id not in (1, 2):
                continue
            y = F.linear(
                F.silu(F.linear(hidden[token_id], gate[expert_id]))
                * F.linear(hidden[token_id], up[expert_id]),
                down[expert_id],
            )
            expected[token_id] += weights[token_id, route_id] * y

    expanded = torch.zeros_like(hidden)
    expanded.index_add_(0, result.token_ids, result.output)
    torch.testing.assert_close(expanded, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result.token_ids, torch.arange(4))
    assert result.active_experts == 2
    assert result.routed_pairs == 6
    assert result.compact_tokens == 4
    assert result.single_token_experts == 0
    assert result.small_experts == 2
    assert result.max_tokens_per_expert == 3
    assert result.workspace_allocations == 1
    assert result.input_bytes > 0
    assert result.output_bytes == 4 * hidden.shape[1] * hidden.element_size()
    assert result.compute_ms > 0
    assert result.gate_up_ms > 0
    assert result.down_ms > 0

    second = executor.execute(
        store=layer,
        hidden_states=hidden,
        expert_ids=expert_ids,
        gate_weights=weights,
        active_experts=(1, 2),
        route_counts={1: 3, 2: 3},
        profile=True,
    )
    assert second.workspace_allocations == 0
    assert executor.workspace_allocations == 1
    executor.close()


def test_inplace_pointwise_matches_baseline_in_bfloat16() -> None:
    torch.manual_seed(23)
    store = PinnedExpertStore(dtype=torch.bfloat16, pin_memory=False)
    layer = store.register_layer(
        layer_id=0,
        num_experts=4,
        hidden_size=32,
        intermediate_size=16,
    )
    for expert_id in range(4):
        layer.put(expert_id, "gate_proj", torch.randn(16, 32))
        layer.put(expert_id, "up_proj", torch.randn(16, 32))
        layer.put(expert_id, "down_proj", torch.randn(32, 16))

    hidden = torch.randn(12, 32, dtype=torch.bfloat16)
    expert_ids = torch.randint(0, 4, (12, 2))
    weights = torch.rand(12, 2, dtype=torch.bfloat16)
    weights /= weights.sum(dim=-1, keepdim=True)
    active_experts = tuple(range(4))
    route_counts = {
        expert_id: int((expert_ids == expert_id).sum())
        for expert_id in active_experts
    }

    outputs = []
    for pointwise_mode in ("baseline", "inplace"):
        executor = CPUExpertExecutor(
            num_threads=1,
            strategy="serial",
            pointwise_mode=pointwise_mode,
        )
        result = executor.execute(
            store=layer,
            hidden_states=hidden,
            expert_ids=expert_ids,
            gate_weights=weights,
            active_experts=active_experts,
            route_counts=route_counts,
            profile=False,
        )
        outputs.append(result)
        executor.close()

    torch.testing.assert_close(outputs[1].token_ids, outputs[0].token_ids)
    torch.testing.assert_close(outputs[1].output, outputs[0].output, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compact_async_output_survives_cross_layer_reuse_without_inner_sync() -> None:
    torch.manual_seed(31)
    hidden_size = 16
    intermediate_size = 8
    store = PinnedExpertStore(dtype=torch.float32, pin_memory=True)
    layer = store.register_layer(
        layer_id=0,
        num_experts=3,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    gate = torch.randn(3, intermediate_size, hidden_size)
    up = torch.randn(3, intermediate_size, hidden_size)
    down = torch.randn(3, hidden_size, intermediate_size)
    for expert_id in range(3):
        layer.put(expert_id, "gate_proj", gate[expert_id])
        layer.put(expert_id, "up_proj", up[expert_id])
        layer.put(expert_id, "down_proj", down[expert_id])

    hidden_host = torch.randn(7, hidden_size)
    hidden = hidden_host.to("cuda")
    expert_ids = torch.tensor(
        [[0, 1], [1, 2], [0, 2], [2, 1], [0, 0], [2, 0], [1, 2]],
        device="cuda",
    )
    weights_host = torch.rand(7, 2)
    weights_host /= weights_host.sum(dim=-1, keepdim=True)
    active_experts = (1, 2)
    route_counts = {
        expert_id: int((expert_ids == expert_id).sum().item())
        for expert_id in active_experts
    }
    route_packet = np.column_stack(
        (
            expert_ids.cpu().numpy().reshape(-1),
            np.zeros(expert_ids.numel(), dtype=np.float32),
            weights_host.numpy().reshape(-1),
        )
    )

    expected = torch.zeros_like(hidden_host)
    expert_ids_host = expert_ids.cpu()
    for token_id in range(hidden_host.shape[0]):
        for route_id in range(expert_ids_host.shape[1]):
            expert_id = int(expert_ids_host[token_id, route_id])
            if expert_id not in active_experts:
                continue
            y = F.linear(
                F.silu(F.linear(hidden_host[token_id], gate[expert_id]))
                * F.linear(hidden_host[token_id], up[expert_id]),
                down[expert_id],
            )
            expected[token_id] += weights_host[token_id, route_id] * y

    executor = CPUExpertExecutor(
        num_threads=2,
        strategy="expert_parallel",
        pointwise_mode="inplace",
    )
    outputs = []
    for iteration in range(12):
        prepared = executor.prepare_compact(
            store=layer,
            hidden_states=hidden,
            expert_ids=expert_ids,
            route_packet_host=route_packet,
            active_experts=active_experts,
            route_counts=route_counts,
            profile=False,
            workspace_scope=iteration % 2,
        )
        host_result = executor.execute_prepared(prepared)
        result = executor.complete(prepared, host_result)
        output = torch.zeros_like(hidden)
        output.index_add_(0, result.token_ids, result.output)
        outputs.append(output)

    torch.cuda.synchronize()
    for output in outputs:
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)
    assert executor.workspace_allocations == 2
    executor.close()
