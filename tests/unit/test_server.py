from __future__ import annotations

import asyncio
import json
import threading
import time

import torch
from fastapi.testclient import TestClient

from oled_moe.server import ServerSettings, create_app
from oled_moe.server.service import LocalChatService


class _Tokenizer:
    eos_token_id = 99

    def __init__(self) -> None:
        self.messages: list[list[dict[str, str]]] = []

    def apply_chat_template(self, messages, **_kwargs):
        self.messages.append(messages)
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def decode(self, token_ids, **_kwargs):
        pieces = {10: "hel", 11: "lo", 12: " 世界", 13: "!"}
        return "".join(pieces[token_id] for token_id in token_ids)


class _Engine:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.tokenizer = _Tokenizer()
        self.delay = delay
        self.closed = False
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def generate_ids(self, _input_ids, *, on_block, **_kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            on_block(torch.tensor([10], dtype=torch.long))
            on_block(torch.tensor([11, 12], dtype=torch.long))
            return torch.tensor([[1, 2, 3, 10, 11, 12]], dtype=torch.long)
        finally:
            with self._lock:
                self.active -= 1

    def close(self) -> None:
        self.closed = True


def _settings() -> ServerSettings:
    return ServerSettings(
        served_model_name="test-model",
        default_max_tokens=3,
        default_block_length=2,
        max_total_tokens=32,
    )


def _data_events(response) -> list[str]:
    return [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]


def test_stream_emits_one_sse_event_per_completed_block() -> None:
    engine = _Engine()
    app = create_app(_settings(), lambda: engine)
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": True,
        "max_tokens": 3,
        "block_length": 2,
        "stream_options": {"include_usage": True},
    }

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _data_events(response)
    assert events[-1] == "[DONE]"
    payloads = [json.loads(event) for event in events[:-1]]
    block_payloads = [payload for payload in payloads if "oled_block_index" in payload]
    assert [payload["oled_block_index"] for payload in block_payloads] == [0, 1]
    assert [payload["choices"][0]["delta"]["content"] for payload in block_payloads] == [
        "hel",
        "lo 世界",
    ]
    assert block_payloads[-1]["oled_text"] == "hello 世界"
    assert payloads[-2]["choices"][0]["finish_reason"] == "length"
    assert payloads[-1]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 3,
        "total_tokens": 6,
    }
    assert engine.closed


def test_non_stream_response_and_full_message_history() -> None:
    engine = _Engine()
    app = create_app(_settings(), lambda: engine)
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "two"},
    ]

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": messages, "max_tokens": 3},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"] == {
        "role": "assistant",
        "content": "hello 世界",
    }
    assert engine.tokenizer.messages == [messages]


def test_semantic_validation_uses_openai_error_shape() -> None:
    engine = _Engine()
    app = create_app(_settings(), lambda: engine)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "wrong-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_single_worker_serializes_generation() -> None:
    engine = _Engine(delay=0.03)
    service = LocalChatService(lambda: engine, max_total_tokens=32)

    async def exercise() -> None:
        await service.start()
        prepared = await service.prepare(
            [{"role": "user", "content": "hi"}], max_tokens=3
        )
        await asyncio.gather(
            service.generate(prepared, max_tokens=3, block_length=2),
            service.generate(prepared, max_tokens=3, block_length=2),
        )
        await service.close()

    asyncio.run(exercise())
    assert engine.max_active == 1
    assert engine.closed
