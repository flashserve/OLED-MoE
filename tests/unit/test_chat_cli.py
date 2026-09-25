from __future__ import annotations

import asyncio
import io

import torch

from oled_moe.cli.chat import interactive_chat
from oled_moe.server.service import LocalChatService


class _Tokenizer:
    eos_token_id = 99

    def __init__(self) -> None:
        self.messages: list[list[dict[str, str]]] = []

    def apply_chat_template(self, messages, **_kwargs):
        self.messages.append([dict(message) for message in messages])
        return torch.tensor([[1, 2]], dtype=torch.long)

    def decode(self, token_ids, **_kwargs):
        pieces = {10: "hel", 11: "lo"}
        return "".join(pieces[token_id] for token_id in token_ids)


class _Engine:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.closed = False

    def generate_ids(self, _input_ids, *, on_block, **_kwargs):
        on_block(torch.tensor([10], dtype=torch.long))
        on_block(torch.tensor([11], dtype=torch.long))
        return torch.tensor([[1, 2, 10, 11]], dtype=torch.long)

    def close(self) -> None:
        self.closed = True


def test_interactive_chat_streams_and_manages_history() -> None:
    engine = _Engine()
    service = LocalChatService(lambda: engine, max_total_tokens=64)
    commands = iter(["first", "second", "/clear", "third", "/exit"])
    output = io.StringIO()

    async def exercise() -> None:
        await service.start()
        try:
            result = await interactive_chat(
                service,
                max_tokens=2,
                block_length=1,
                system_prompt="Be concise.",
                input_fn=lambda: next(commands),
                output=output,
            )
            assert result == 0
        finally:
            await service.close()

    asyncio.run(exercise())

    assert output.getvalue().count("Assistant> hello") == 3
    assert "[conversation cleared]" in output.getvalue()
    assert engine.tokenizer.messages == [
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "first"},
        ],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "second"},
        ],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "third"},
        ],
    ]
    assert engine.closed
