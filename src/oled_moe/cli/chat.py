from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from typing import TextIO

from oled_moe.server.service import LocalChatService, TextBlock

from .serve import add_engine_arguments, build_config


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_engine_arguments(parser)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument(
        "--system-prompt",
        default="",
        help="optional system message kept across turns and /clear",
    )


def _validate(args: argparse.Namespace) -> None:
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")
    if args.block_length <= 0:
        raise ValueError("block-length must be positive")
    if args.block_length > args.max_tokens:
        raise ValueError("block-length must not exceed max-tokens")
    if args.max_total_tokens <= args.max_tokens:
        raise ValueError("max-total-tokens must be larger than max-tokens")


def _new_history(system_prompt: str) -> list[dict[str, str]]:
    if system_prompt:
        return [{"role": "system", "content": system_prompt}]
    return []


async def interactive_chat(
    service: LocalChatService,
    *,
    max_tokens: int,
    block_length: int,
    system_prompt: str = "",
    input_fn: Callable[[], str] = input,
    output: TextIO = sys.stdout,
) -> int:
    history = _new_history(system_prompt)
    print("OLED-MoE chat ready. Commands: /clear, /help, /exit", file=output)

    while True:
        print("You> ", end="", file=output, flush=True)
        try:
            question = input_fn().strip()
        except EOFError:
            print("\nBye.", file=output)
            return 0
        except KeyboardInterrupt:
            print("\nBye.", file=output)
            return 0

        if not question:
            continue
        command = question.casefold()
        if command in {"/exit", "/quit"}:
            print("Bye.", file=output)
            return 0
        if command == "/clear":
            history = _new_history(system_prompt)
            print("[conversation cleared]", file=output)
            continue
        if command == "/help":
            print(
                "/clear  clear conversation history\n"
                "/exit   close the model and exit",
                file=output,
            )
            continue

        pending = [*history, {"role": "user", "content": question}]
        try:
            prepared = await service.prepare(pending, max_tokens=max_tokens)
        except ValueError as exc:
            print(f"[error] {exc}", file=output)
            continue

        print("Assistant> ", end="", file=output, flush=True)

        def render(block: TextBlock) -> None:
            if block.delta:
                output.write(block.delta)
                output.flush()

        try:
            result = await service.generate(
                prepared,
                max_tokens=max_tokens,
                block_length=block_length,
                on_block=render,
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary must close the model cleanly
            print(
                f"\n[error] generation failed; closing the session: {exc}",
                file=output,
            )
            return 1

        print(file=output, flush=True)
        history = [
            *pending,
            {"role": "assistant", "content": result.text},
        ]


def run(args: argparse.Namespace) -> int:
    _validate(args)

    from oled_moe.engine import Engine

    config = build_config(args)

    def load_engine() -> Engine:
        return Engine.from_pretrained(
            args.model,
            config=config,
            device=args.device,
            threshold=args.threshold,
            dinfer_path=args.dinfer_path,
        )

    service = LocalChatService(load_engine, max_total_tokens=args.max_total_tokens)

    async def session() -> int:
        print(f"Loading model from {args.model} ...", flush=True)
        await service.start()
        try:
            return await interactive_chat(
                service,
                max_tokens=args.max_tokens,
                block_length=args.block_length,
                system_prompt=args.system_prompt,
            )
        finally:
            await service.close()

    try:
        return asyncio.run(session())
    except KeyboardInterrupt:
        print("\nInterrupted. The active GPU request was allowed to finish safely.")
        return 130
