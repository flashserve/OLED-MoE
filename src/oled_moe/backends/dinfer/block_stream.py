from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch


def _decode_argument(
    args: tuple[Any, ...], kwargs: dict[str, Any], position: int, name: str
) -> Any:
    if len(args) > position:
        return args[position]
    try:
        return kwargs[name]
    except KeyError as exc:
        raise RuntimeError(f"dInfer decode no longer exposes {name!r}") from exc


@contextmanager
def completed_block_callback(
    generator: object,
    *,
    prompt_length: int,
    gen_length: int,
    on_block: Callable[[torch.Tensor], None] | None,
) -> Iterator[None]:
    """Emit the generated slice immediately after dInfer finishes each block.

    dInfer aligns the full sequence to a block boundary, and the first decoded
    block may overlap the prompt. The callback therefore sees only tokens in
    ``[prompt_length, prompt_length + gen_length)``.
    """

    if on_block is None:
        yield
        return

    runner = getattr(generator, "block_runner", None)
    original_decode = getattr(runner, "decode", None)
    if runner is None or not callable(original_decode):
        raise RuntimeError("this dInfer generator does not support block streaming")

    generation_end = prompt_length + gen_length

    def decode_and_emit(*args: Any, **kwargs: Any) -> Any:
        result = original_decode(*args, **kwargs)
        token_array = _decode_argument(args, kwargs, 2, "x")
        block_loc = _decode_argument(args, kwargs, 5, "block_loc")
        start = max(prompt_length, int(block_loc.start))
        end = min(generation_end, int(block_loc.end))
        if start < end:
            tokens = token_array[0, start:end].detach().to("cpu").clone()
            on_block(tokens)
        return result

    runner.decode = decode_and_emit
    try:
        yield
    finally:
        runner.decode = original_decode
