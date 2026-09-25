from __future__ import annotations

import argparse

from oled_moe import (
    CacheConfig,
    CPUConfig,
    LayerCapacityConfig,
    OLEDConfig,
    PlacementConfig,
)


def add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="local model checkpoint directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dinfer-path")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--cache-size", type=int, default=40)
    parser.add_argument(
        "--cache-policy",
        choices=("lru", "lru_observed", "lfu", "priority"),
        default="priority",
    )
    parser.add_argument(
        "--layer-capacity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply adaptive per-layer cache capacity (default: enabled)",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("gpu_only", "cpu_only", "static_gpu", "static_hybrid", "cooperative"),
        default="gpu_only",
    )
    parser.add_argument("--static-gpu-experts-per-layer", type=int, default=0)
    parser.add_argument("--cpu-miss-ratio", type=float, default=0.5)
    parser.add_argument("--cpu-min-dispatch-experts", type=int, default=1)
    parser.add_argument("--cpu-num-threads", type=int, default=16)
    parser.add_argument(
        "--cpu-execution-strategy",
        choices=("serial", "expert_parallel"),
        default="expert_parallel",
    )
    parser.add_argument(
        "--cpu-pointwise-mode", choices=("baseline", "inplace"), default="inplace"
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_engine_arguments(parser)
    parser.add_argument("--served-model-name", default="oled-moe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-max-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--log-level", default="info")


def _validate(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if args.default_max_tokens <= 0:
        raise ValueError("default-max-tokens must be positive")
    if args.block_length <= 0:
        raise ValueError("block-length must be positive")
    if args.block_length > args.default_max_tokens:
        raise ValueError("block-length must not exceed default-max-tokens")
    if args.max_total_tokens <= args.default_max_tokens:
        raise ValueError("max-total-tokens must be larger than default-max-tokens")


def build_config(args: argparse.Namespace) -> OLEDConfig:
    return OLEDConfig(
        cache=CacheConfig(
            capacity_per_layer=args.cache_size,
            policy=args.cache_policy,
        ),
        layer_capacity=LayerCapacityConfig(
            enabled=args.layer_capacity,
            apply=args.layer_capacity,
        ),
        placement=PlacementConfig(
            mode=args.execution_mode,
            static_gpu_experts_per_layer=args.static_gpu_experts_per_layer,
            cpu_miss_ratio=args.cpu_miss_ratio,
            cpu_min_dispatch_experts=args.cpu_min_dispatch_experts,
        ),
        cpu=CPUConfig(
            num_threads=args.cpu_num_threads,
            strategy=args.cpu_execution_strategy,
            pointwise_mode=args.cpu_pointwise_mode,
        ),
    )


def run(args: argparse.Namespace) -> int:
    _validate(args)

    import uvicorn

    from oled_moe.engine import Engine
    from oled_moe.server import ServerSettings, create_app

    config = build_config(args)
    settings = ServerSettings(
        served_model_name=args.served_model_name,
        default_max_tokens=args.default_max_tokens,
        default_block_length=args.block_length,
        max_total_tokens=args.max_total_tokens,
    )

    def load_engine() -> Engine:
        return Engine.from_pretrained(
            args.model,
            config=config,
            device=args.device,
            threshold=args.threshold,
            dinfer_path=args.dinfer_path,
        )

    app = create_app(settings, load_engine)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        log_level=args.log_level,
    )
    return 0
