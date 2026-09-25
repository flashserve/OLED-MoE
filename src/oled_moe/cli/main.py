from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import benchmark, chat, cpu_microbench, serve


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oled-moe")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark_parser = subparsers.add_parser("benchmark", help="run an OLED-MoE benchmark")
    benchmark.add_arguments(benchmark_parser)
    cpu_parser = subparsers.add_parser(
        "cpu-microbench",
        help="measure the standalone CPU expert executor without loading a model",
    )
    cpu_microbench.add_arguments(cpu_parser)
    serve_parser = subparsers.add_parser(
        "serve", help="serve local OpenAI-compatible block-streaming chat"
    )
    serve.add_arguments(serve_parser)
    chat_parser = subparsers.add_parser(
        "chat", help="open a local block-streaming terminal conversation"
    )
    chat.add_arguments(chat_parser)
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        return benchmark.run(args)
    if args.command == "cpu-microbench":
        return cpu_microbench.run(args)
    if args.command == "serve":
        return serve.run(args)
    if args.command == "chat":
        return chat.run(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
