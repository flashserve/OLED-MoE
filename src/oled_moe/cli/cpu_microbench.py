from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from oled_moe.executor import CPUExpertExecutor
from oled_moe.memory import PinnedExpertStore


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--active-experts", type=int, default=58)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--cpu-threads",
        default="1,2,4,8,16,26",
        help=(
            "comma-separated thread budgets: expert workers for expert_parallel or "
            "intra-op threads for serial"
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=("serial", "expert_parallel"),
        default="expert_parallel",
    )
    parser.add_argument(
        "--pointwise-mode",
        choices=("baseline", "inplace"),
        default="inplace",
    )
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--output", help="optional JSON result path")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = percentile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _mean(results: list[Any], field: str) -> float:
    return statistics.mean(float(getattr(result, field)) for result in results)


def _thread_counts(value: str) -> list[int]:
    counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("cpu-threads must contain positive comma-separated integers")
    return counts


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args: argparse.Namespace) -> int:
    if args.iters <= 0 or args.warmup_iters < 0:
        raise ValueError("iters must be positive and warmup-iters must be non-negative")
    if args.num_experts <= 0 or not 0 < args.active_experts <= args.num_experts:
        raise ValueError("active-experts must be in [1, num-experts]")
    if args.tokens <= 0 or args.top_k <= 0:
        raise ValueError("tokens and top-k must be positive")
    routed_pairs = args.tokens * args.top_k
    if args.active_experts > routed_pairs:
        raise ValueError("active-experts cannot exceed tokens * top-k")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    pin_memory = not args.no_pin_memory and device.type == "cuda"
    torch.manual_seed(args.seed)

    store = PinnedExpertStore(dtype=dtype, pin_memory=pin_memory)
    layer = store.register_layer(
        layer_id=0,
        num_experts=args.num_experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
    )
    gate = torch.empty((args.intermediate_size, args.hidden_size), dtype=dtype).uniform_(
        -0.01, 0.01
    )
    up = torch.empty_like(gate).uniform_(-0.01, 0.01)
    down = torch.empty((args.hidden_size, args.intermediate_size), dtype=dtype).uniform_(
        -0.01, 0.01
    )
    for expert_id in range(args.active_experts):
        layer.put(expert_id, "gate_proj", gate)
        layer.put(expert_id, "up_proj", up)
        layer.put(expert_id, "down_proj", down)

    flat_experts = torch.arange(routed_pairs, dtype=torch.long) % args.active_experts
    permutation = torch.randperm(routed_pairs)
    expert_ids_cpu = flat_experts[permutation].reshape(args.tokens, args.top_k)
    unique, counts = torch.unique(expert_ids_cpu, sorted=True, return_counts=True)
    active_experts = tuple(int(value) for value in unique.tolist())
    route_counts = {
        int(expert): int(count)
        for expert, count in zip(unique.tolist(), counts.tolist(), strict=True)
    }
    expert_ids = expert_ids_cpu.to(device)
    gate_weights = torch.rand((args.tokens, args.top_k), device=device, dtype=torch.float32)
    gate_weights /= gate_weights.sum(dim=-1, keepdim=True)
    hidden_states = torch.randn(
        (args.tokens, args.hidden_size),
        device=device,
        dtype=dtype,
    )

    thread_results: list[dict[str, Any]] = []
    for threads in _thread_counts(args.cpu_threads):
        executor = CPUExpertExecutor(
            num_threads=threads,
            strategy=args.strategy,
            pointwise_mode=args.pointwise_mode,
        )
        for _ in range(args.warmup_iters):
            executor.execute(
                store=layer,
                hidden_states=hidden_states,
                expert_ids=expert_ids,
                gate_weights=gate_weights,
                active_experts=active_experts,
                route_counts=route_counts,
                profile=False,
            )
        _synchronize(device)

        wall_ms: list[float] = []
        measured = []
        for _ in range(args.iters):
            _synchronize(device)
            started = time.perf_counter_ns()
            result = executor.execute(
                store=layer,
                hidden_states=hidden_states,
                expert_ids=expert_ids,
                gate_weights=gate_weights,
                active_experts=active_experts,
                route_counts=route_counts,
                profile=True,
            )
            _synchronize(device)
            wall_ms.append((time.perf_counter_ns() - started) / 1e6)
            measured.append(result)

        record = {
            "cpu_threads": threads,
            "strategy": args.strategy,
            "pointwise_mode": args.pointwise_mode,
            "latency_ms": {
                "mean": statistics.mean(wall_ms),
                "p50": _percentile(wall_ms, 0.50),
                "p90": _percentile(wall_ms, 0.90),
                "min": min(wall_ms),
                "max": max(wall_ms),
            },
            "stage_mean_ms": {
                "input": _mean(measured, "input_copy_ms"),
                "route": _mean(measured, "route_ms"),
                "compute": _mean(measured, "compute_ms"),
                "gate_up": _mean(measured, "gate_up_ms"),
                "activation": _mean(measured, "activation_ms"),
                "down": _mean(measured, "down_ms"),
                "scatter": _mean(measured, "scatter_ms"),
                "output": _mean(measured, "output_copy_ms"),
            },
            "workspace_allocations": executor.workspace_allocations,
        }
        thread_results.append(record)
        print("[OLED CPU microbench] " + json.dumps(record, sort_keys=True))
        executor.close()

    route_values = list(route_counts.values())
    payload = {
        "configuration": {
            "device": str(device),
            "dtype": str(dtype),
            "pin_memory": pin_memory,
            "num_experts": args.num_experts,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "tokens": args.tokens,
            "top_k": args.top_k,
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "seed": args.seed,
            "strategy": args.strategy,
            "pointwise_mode": args.pointwise_mode,
        },
        "routing": {
            "routed_pairs": routed_pairs,
            "active_experts": len(active_experts),
            "mean_tokens_per_expert": statistics.mean(route_values),
            "max_tokens_per_expert": max(route_values),
            "single_token_experts": sum(count == 1 for count in route_values),
            "small_experts": sum(count <= 4 for count in route_values),
        },
        "results": thread_results,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[OLED CPU microbench report] {output_path.resolve()}")
    return 0
