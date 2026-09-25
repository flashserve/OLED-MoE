from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch

from oled_moe import (
    CacheConfig,
    CPUConfig,
    Engine,
    LayerCapacityConfig,
    OLEDConfig,
    PlacementConfig,
    ProfilingConfig,
    TransferConfig,
)
from oled_moe.observability import write_performance_report

DEFAULT_PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. After that, she can run "
    "6 kilometers per hour. How many kilometers can she run in 8 hours?"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dinfer-path")
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--cache-size", type=int, default=100)
    parser.add_argument(
        "--cache-policy",
        choices=("lru", "lru_observed", "lfu", "priority"),
        default="lru",
    )
    parser.add_argument("--priority-confidence-power", type=float, default=1.0)
    parser.add_argument("--priority-gate-power", type=float, default=1.0)
    parser.add_argument(
        "--priority-update-interval",
        type=int,
        default=1,
        help="priority refresh period per layer; 1 computes every layer call",
    )
    parser.add_argument(
        "--selective-priority",
        action="store_true",
        help="only compute scheduled priority updates for high-opportunity layers",
    )
    parser.add_argument(
        "--no-selective-priority",
        dest="selective_priority",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(selective_priority=False)
    parser.add_argument("--priority-layer-fraction", type=float, default=0.5)
    parser.add_argument("--priority-utilization-ema-decay", type=float, default=0.9)
    parser.add_argument("--priority-min-opportunity", type=float, default=0.02)
    parser.add_argument("--priority-hysteresis", type=float, default=0.01)
    parser.add_argument("--layer-capacity-dry-run", action="store_true")
    parser.add_argument("--layer-capacity-apply", action="store_true")
    parser.add_argument("--layer-capacity-warmup-forwards", type=int, default=8)
    parser.add_argument("--layer-capacity-update-interval", type=int, default=8)
    parser.add_argument("--layer-capacity-steady-update-interval", type=int, default=30)
    parser.add_argument("--layer-capacity-convergence-patience", type=int, default=3)
    parser.add_argument("--layer-capacity-convergence-moved-ratio", type=float, default=0.005)
    parser.add_argument("--layer-capacity-ema-decay", type=float, default=0.8)
    parser.add_argument("--layer-capacity-activation-power", type=float, default=1.0)
    parser.add_argument("--layer-capacity-utilization-power", type=float, default=1.0)
    parser.add_argument("--layer-capacity-utilization-floor", type=float, default=0.05)
    parser.add_argument("--layer-capacity-min-ratio", type=float, default=0.25)
    parser.add_argument("--layer-capacity-max-ratio", type=float, default=2.0)
    parser.add_argument("--layer-capacity-max-moved-ratio", type=float, default=0.10)
    parser.add_argument("--layer-capacity-hysteresis", type=float, default=0.01)
    parser.add_argument(
        "--execution-mode",
        choices=("gpu_only", "cpu_only", "static_gpu", "static_hybrid", "cooperative"),
        default="gpu_only",
    )
    parser.add_argument("--static-gpu-experts-per-layer", type=int, default=0)
    parser.add_argument(
        "--cpu-miss-ratio",
        type=float,
        default=0.5,
        help="fraction of cooperative cache-miss experts assigned to CPU execution",
    )
    parser.add_argument(
        "--cpu-min-dispatch-experts",
        type=int,
        default=1,
        help=(
            "minimum CPU experts per cooperative dispatch; larger values "
            "amortize route, copy, and thread-pool overhead"
        ),
    )
    parser.add_argument(
        "--transfer-high-priority",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "send high-priority misses through H2D/GPU; use "
            "--no-transfer-high-priority to transfer the lowest-priority misses"
        ),
    )
    parser.add_argument("--cpu-num-threads", type=int, default=16)
    parser.add_argument(
        "--cpu-execution-strategy",
        choices=("serial", "expert_parallel"),
        default="expert_parallel",
    )
    parser.add_argument(
        "--cpu-pointwise-mode",
        choices=("baseline", "inplace"),
        default="inplace",
    )
    parser.add_argument(
        "--cpu-compact-io",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "pack only CPU-routed tokens and overlap compact CPU I/O; use "
            "--no-cpu-compact-io for the dense reference path"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--no-copy-stream", action="store_true")
    parser.add_argument("--no-transfer-overlap", action="store_true")
    parser.add_argument(
        "--split-weight-transfer",
        action="store_true",
        help="diagnostic control: submit w13 and w2 as two H2D copies per expert",
    )
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="collect synchronized host/CUDA timings and write a dashboard",
    )
    parser.add_argument("--report-dir", default="runs")


def _prompt_ids(tokenizer: object, prompt: str, device: torch.device) -> torch.Tensor:
    wrapped = (
        "<role>SYSTEM</role>detailed thinking off<|role_end|>"
        f"<role>HUMAN</role>{prompt}<|role_end|><role>ASSISTANT</role>"
    )
    ids = tokenizer(wrapped)["input_ids"]
    return torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)


def _generated_token_count(
    output: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    block_length: int,
    gen_length: int,
) -> int:
    prompt_length = int(input_ids.shape[-1])
    aligned_prompt_length = ((prompt_length + block_length - 1) // block_length) * block_length
    return min(
        gen_length,
        max(0, int(output.shape[-1]) - aligned_prompt_length),
    )


def _hardware(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "device_name": properties.name,
        "device_total_memory_gib": properties.total_memory / 1024**3,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def run(args: argparse.Namespace) -> int:
    if args.layer_capacity_dry_run and args.layer_capacity_apply:
        raise ValueError("choose either layer capacity dry-run or apply")
    if args.runs <= 0 or args.warmup_runs < 0:
        raise ValueError("runs must be positive and warmup-runs must be non-negative")
    config = OLEDConfig(
        cache=CacheConfig(
            capacity_per_layer=args.cache_size,
            policy=args.cache_policy,
            confidence_power=args.priority_confidence_power,
            gate_power=args.priority_gate_power,
            update_interval=args.priority_update_interval,
            selective_priority=args.selective_priority,
            priority_layer_fraction=args.priority_layer_fraction,
            utilization_ema_decay=args.priority_utilization_ema_decay,
            min_priority_opportunity=args.priority_min_opportunity,
            priority_hysteresis=args.priority_hysteresis,
        ),
        transfer=TransferConfig(
            use_copy_stream=not args.no_copy_stream,
            overlap_compute=not args.no_transfer_overlap,
            coalesce_expert_weights=not args.split_weight_transfer,
        ),
        layer_capacity=LayerCapacityConfig(
            enabled=args.layer_capacity_dry_run or args.layer_capacity_apply,
            apply=args.layer_capacity_apply,
            warmup_forwards=args.layer_capacity_warmup_forwards,
            update_interval=args.layer_capacity_update_interval,
            steady_update_interval=args.layer_capacity_steady_update_interval,
            convergence_patience=args.layer_capacity_convergence_patience,
            convergence_moved_ratio=args.layer_capacity_convergence_moved_ratio,
            ema_decay=args.layer_capacity_ema_decay,
            activation_power=args.layer_capacity_activation_power,
            utilization_power=args.layer_capacity_utilization_power,
            utilization_floor=args.layer_capacity_utilization_floor,
            min_capacity_ratio=args.layer_capacity_min_ratio,
            max_capacity_ratio=args.layer_capacity_max_ratio,
            max_slots_moved_ratio=args.layer_capacity_max_moved_ratio,
            capacity_hysteresis=args.layer_capacity_hysteresis,
        ),
        profiling=ProfilingConfig(enabled=args.profile),
        placement=PlacementConfig(
            mode=args.execution_mode,
            static_gpu_experts_per_layer=args.static_gpu_experts_per_layer,
            cpu_miss_ratio=args.cpu_miss_ratio,
            cpu_min_dispatch_experts=args.cpu_min_dispatch_experts,
            transfer_high_priority=args.transfer_high_priority,
        ),
        cpu=CPUConfig(
            num_threads=args.cpu_num_threads,
            strategy=args.cpu_execution_strategy,
            pointwise_mode=args.cpu_pointwise_mode,
            compact_io=args.cpu_compact_io,
        ),
        pin_memory=not args.no_pin_memory,
    )
    resolved_config = {
        "cache_size": args.cache_size,
        "cache_policy": config.cache.policy,
        "priority_confidence_power": config.cache.confidence_power,
        "priority_gate_power": config.cache.gate_power,
        "priority_update_interval": config.cache.update_interval,
        "selective_priority": config.cache.selective_priority,
        "priority_layer_fraction": config.cache.priority_layer_fraction,
        "priority_utilization_ema_decay": config.cache.utilization_ema_decay,
        "priority_min_opportunity": config.cache.min_priority_opportunity,
        "priority_hysteresis": config.cache.priority_hysteresis,
        "layer_capacity_dry_run": (
            config.layer_capacity.enabled and not config.layer_capacity.apply
        ),
        "layer_capacity_apply": config.layer_capacity.apply,
        "layer_capacity_warmup_forwards": config.layer_capacity.warmup_forwards,
        "layer_capacity_update_interval": config.layer_capacity.update_interval,
        "layer_capacity_ema_decay": config.layer_capacity.ema_decay,
        "layer_capacity_activation_power": config.layer_capacity.activation_power,
        "layer_capacity_utilization_power": config.layer_capacity.utilization_power,
        "layer_capacity_utilization_floor": config.layer_capacity.utilization_floor,
        "layer_capacity_min_ratio": config.layer_capacity.min_capacity_ratio,
        "layer_capacity_max_ratio": config.layer_capacity.max_capacity_ratio,
        "layer_capacity_max_moved_ratio": config.layer_capacity.max_slots_moved_ratio,
        "layer_capacity_hysteresis": config.layer_capacity.capacity_hysteresis,
        "execution_mode": config.placement.mode,
        "static_gpu_experts_per_layer": (config.placement.static_gpu_experts_per_layer),
        "cpu_miss_ratio": config.placement.cpu_miss_ratio,
        "cpu_min_dispatch_experts": (
            config.placement.cpu_min_dispatch_experts
        ),
        "transfer_high_priority": config.placement.transfer_high_priority,
        "cpu_num_threads": config.cpu.num_threads,
        "cpu_execution_strategy": config.cpu.strategy,
        "cpu_pointwise_mode": config.cpu.pointwise_mode,
        "cpu_compact_io": config.cpu.compact_io,
        "pin_memory": config.pin_memory,
        "copy_stream": config.transfer.use_copy_stream,
        "transfer_overlap": config.transfer.overlap_compute,
        "coalesced_expert_weights": config.transfer.coalesce_expert_weights,
        "profiling": config.profiling.enabled,
        "warmup_runs": args.warmup_runs,
        "runs": args.runs,
        "batch_size": 1,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "threshold": args.threshold,
        "seed": args.seed,
    }
    print("[OLED] resolved_config=" + json.dumps(resolved_config, sort_keys=True))
    engine: Engine | None = None
    try:
        load_start = time.perf_counter()
        engine = Engine.from_pretrained(
            args.model,
            config=config,
            device=args.device,
            threshold=args.threshold,
            dinfer_path=args.dinfer_path,
        )
        torch.cuda.synchronize(engine.session.device)
        load_seconds = time.perf_counter() - load_start
        input_ids = _prompt_ids(engine.tokenizer, args.prompt, engine.session.device)
        print(
            f"[OLED] model_loaded_s={load_seconds:.3f} "
            f"host_expert_gib={engine.runtime.store.allocated_bytes / 1024**3:.3f}"
        )

        for warmup_id in range(args.warmup_runs):
            torch.manual_seed(args.seed + warmup_id)
            engine.generate_ids(
                input_ids,
                gen_length=args.gen_length,
                block_length=args.block_length,
            )
        torch.cuda.synchronize(engine.session.device)
        engine.runtime.reset_metrics()
        torch.cuda.reset_peak_memory_stats(engine.session.device)

        records: list[dict[str, float | int]] = []
        last_output: torch.Tensor | None = None
        for run_id in range(1, args.runs + 1):
            torch.manual_seed(args.seed + args.warmup_runs + run_id)
            previous_forwards = engine.session.generator.num_forwards
            torch.cuda.synchronize(engine.session.device)
            started = time.perf_counter()
            output = engine.generate_ids(
                input_ids,
                gen_length=args.gen_length,
                block_length=args.block_length,
            )
            torch.cuda.synchronize(engine.session.device)
            elapsed = time.perf_counter() - started
            forwards = int(engine.session.generator.num_forwards - previous_forwards)
            output_tokens = _generated_token_count(
                output,
                input_ids,
                block_length=args.block_length,
                gen_length=args.gen_length,
            )
            record = {
                "run": run_id,
                "seconds": elapsed,
                "forwards": forwards,
                "output_tokens": output_tokens,
                "tps": output_tokens / elapsed if elapsed else 0.0,
                "fps": forwards / elapsed if elapsed else 0.0,
                "latency_ms_per_output_token": (
                    elapsed * 1000 / output_tokens if output_tokens else 0.0
                ),
            }
            records.append(record)
            last_output = output
            print("[OLED run] " + json.dumps(record, sort_keys=True))

        runtime_summary = engine.runtime.summary()
        result = {
            "runs": args.runs,
            "batch_size": 1,
            "requested_gen_length": args.gen_length,
            "block_length": args.block_length,
            "mean_seconds": statistics.mean(float(record["seconds"]) for record in records),
            "mean_tps": statistics.mean(float(record["tps"]) for record in records),
            "mean_fps": statistics.mean(float(record["fps"]) for record in records),
            "mean_latency_ms_per_output_token": statistics.mean(
                float(record["latency_ms_per_output_token"]) for record in records
            ),
            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(engine.session.device)
            / 1024**3,
            "runtime": runtime_summary,
            "layer_capacity": engine.runtime.capacity_plan(),
        }
        print("[OLED summary] " + json.dumps(result, sort_keys=True))

        if args.profile:
            timestamp = datetime.now().astimezone()
            report_path = Path(args.report_dir) / timestamp.strftime("%Y%m%d-%H%M%S")
            payload = {
                "created_at": timestamp.isoformat(),
                "model": str(Path(args.model).resolve()),
                "configuration": resolved_config,
                "hardware": _hardware(engine.session.device),
                "model_load_seconds": load_seconds,
                "runs": records,
                "benchmark": result,
                "layers": engine.runtime.metrics(),
            }
            json_path, html_path = write_performance_report(payload, report_path)
            print(
                "[OLED report] "
                + json.dumps(
                    {"json": str(json_path.resolve()), "html": str(html_path.resolve())},
                    sort_keys=True,
                )
            )

        if args.print_output and last_output is not None:
            print(engine.tokenizer.decode(last_output[0], skip_special_tokens=True))
        return 0
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    raise SystemExit(run(parser.parse_args()))
