from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PreparedChat:
    input_ids: torch.Tensor
    prompt_tokens: int


@dataclass(frozen=True)
class TextBlock:
    index: int
    delta: str
    text: str
    token_count: int


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class LocalChatService:
    """Own one model and run every tokenizer/model operation on one worker."""

    def __init__(
        self,
        engine_factory: Callable[[], Any],
        *,
        max_total_tokens: int,
    ) -> None:
        self._engine_factory = engine_factory
        self._max_total_tokens = max_total_tokens
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oled-model")
        self._engine: Any | None = None

    @property
    def ready(self) -> bool:
        return self._engine is not None

    async def start(self) -> None:
        if self._engine is not None:
            return
        loop = asyncio.get_running_loop()
        try:
            self._engine = await loop.run_in_executor(
                self._executor, self._engine_factory
            )
        except BaseException:
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise

    async def close(self) -> None:
        engine, self._engine = self._engine, None
        try:
            if engine is not None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, engine.close)
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)

    async def prepare(
        self, messages: list[dict[str, str]], *, max_tokens: int
    ) -> PreparedChat:
        engine = self._require_engine()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._prepare_sync,
            engine,
            messages,
            max_tokens,
        )

    async def generate(
        self,
        prepared: PreparedChat,
        *,
        max_tokens: int,
        block_length: int,
        on_block: Callable[[TextBlock], None] | None = None,
    ) -> GenerationResult:
        engine = self._require_engine()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._generate_sync,
            engine,
            prepared,
            max_tokens,
            block_length,
            on_block,
        )

    def _require_engine(self) -> Any:
        if self._engine is None:
            raise RuntimeError("model is not loaded")
        return self._engine

    def _prepare_sync(
        self,
        engine: Any,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> PreparedChat:
        tokenizer = engine.tokenizer
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("the chat template must produce one token sequence")
        prompt_tokens = int(input_ids.shape[-1])
        if prompt_tokens + max_tokens > self._max_total_tokens:
            raise ValueError(
                "prompt and completion exceed the configured token limit "
                f"({prompt_tokens} + {max_tokens} > {self._max_total_tokens})"
            )
        return PreparedChat(input_ids=input_ids, prompt_tokens=prompt_tokens)

    @staticmethod
    def _eos_token_ids(engine: Any) -> set[int]:
        eos = getattr(engine.tokenizer, "eos_token_id", None)
        values: set[int] = set()
        if isinstance(eos, int):
            values.add(eos)
        elif eos is not None:
            values.update(int(value) for value in eos)
        decoder = getattr(getattr(engine, "session", None), "generator", None)
        decoder = getattr(decoder, "decoder", None)
        backend_eos = getattr(decoder, "eos_id", None)
        if backend_eos is not None:
            values.add(int(backend_eos))
        return values

    @classmethod
    def _generate_sync(
        cls,
        engine: Any,
        prepared: PreparedChat,
        max_tokens: int,
        block_length: int,
        on_block: Callable[[TextBlock], None] | None,
    ) -> GenerationResult:
        tokenizer = engine.tokenizer
        eos_token_ids = cls._eos_token_ids(engine)
        generated: list[int] = []
        previous_text = ""
        block_index = 0
        saw_eos = False

        def accept_block(tokens: torch.Tensor) -> None:
            nonlocal block_index, previous_text, saw_eos
            if saw_eos or len(generated) >= max_tokens:
                return
            accepted: list[int] = []
            for token_id in tokens.reshape(-1).tolist():
                token_id = int(token_id)
                if token_id in eos_token_ids:
                    saw_eos = True
                    break
                if len(generated) + len(accepted) >= max_tokens:
                    break
                accepted.append(token_id)
            generated.extend(accepted)
            text = tokenizer.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if text.startswith(previous_text):
                delta = text[len(previous_text) :]
            else:
                # The bundled fast tokenizer is prefix-stable with cleanup off.
                # Keep a useful fallback for custom tokenizers instead of dropping text.
                delta = text
            if on_block is not None:
                on_block(
                    TextBlock(
                        index=block_index,
                        delta=delta,
                        text=text,
                        token_count=len(accepted),
                    )
                )
            block_index += 1
            previous_text = text

        engine.generate_ids(
            prepared.input_ids,
            gen_length=max_tokens,
            block_length=block_length,
            on_block=accept_block,
        )
        if block_index == 0:
            raise RuntimeError("dInfer completed without exposing a generated block")
        return GenerationResult(
            text=previous_text,
            prompt_tokens=prepared.prompt_tokens,
            completion_tokens=len(generated),
            finish_reason="stop" if saw_eos else "length",
        )
