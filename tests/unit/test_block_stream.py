from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from oled_moe.backends.dinfer.block_stream import completed_block_callback


@dataclass
class _BlockLoc:
    start: int
    end: int


class _TokenArray:
    def __init__(self, values: list[int]) -> None:
        self.data = torch.tensor([values], dtype=torch.long)

    def __getitem__(self, item):
        return self.data[item]


class _Runner:
    def decode(self, _model, _decoder, x, _cache, _block, block_loc, *_rest):
        x.data[0, block_loc.start : block_loc.end] += 100
        return torch.tensor([True])


class _Generator:
    def __init__(self) -> None:
        self.block_runner = _Runner()


def _decode(runner: _Runner, x: _TokenArray, start: int, end: int) -> None:
    runner.decode(None, None, x, None, None, _BlockLoc(start, end), 0)


def test_callback_clips_prompt_overlap_and_alignment_padding() -> None:
    generator = _Generator()
    tokens = _TokenArray(list(range(16)))
    received: list[torch.Tensor] = []

    with completed_block_callback(
        generator,
        prompt_length=5,
        gen_length=7,
        on_block=received.append,
    ):
        _decode(generator.block_runner, tokens, 4, 8)
        _decode(generator.block_runner, tokens, 8, 16)

    assert [chunk.tolist() for chunk in received] == [
        [105, 106, 107],
        [108, 109, 110, 111],
    ]
    assert sum(chunk.numel() for chunk in received) == 7

    tokens.data.fill_(-1)
    assert received[0].tolist() == [105, 106, 107]


def test_decode_is_restored_after_callback_failure() -> None:
    generator = _Generator()
    tokens = _TokenArray(list(range(8)))
    original_decode = generator.block_runner.decode

    def fail(_tokens: torch.Tensor) -> None:
        raise RuntimeError("consumer failed")

    with pytest.raises(RuntimeError, match="consumer failed"), completed_block_callback(
        generator,
        prompt_length=2,
        gen_length=4,
        on_block=fail,
    ):
        _decode(generator.block_runner, tokens, 0, 4)

    assert generator.block_runner.decode == original_decode


def test_none_callback_leaves_runner_untouched() -> None:
    generator = _Generator()
    tokens = _TokenArray(list(range(8)))
    original_decode = generator.block_runner.decode

    with completed_block_callback(
        generator,
        prompt_length=2,
        gen_length=4,
        on_block=None,
    ):
        assert generator.block_runner.decode == original_decode
        _decode(generator.block_runner, tokens, 0, 4)

    assert generator.block_runner.decode == original_decode
