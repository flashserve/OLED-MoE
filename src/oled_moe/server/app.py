from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .schemas import ChatCompletionRequest
from .service import GenerationResult, LocalChatService, TextBlock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerSettings:
    served_model_name: str = "oled-moe"
    default_max_tokens: int = 128
    default_block_length: int = 16
    max_total_tokens: int = 4096


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _sse(data: object) -> str:
    return f"data: {_json(data)}\n\n"


def _error_payload(message: str, *, code: str, error_type: str) -> dict[str, object]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


def _usage(result: GenerationResult) -> dict[str, int]:
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }


def _resolve_request(
    body: ChatCompletionRequest, settings: ServerSettings
) -> tuple[int, int]:
    if body.model is not None and body.model != settings.served_model_name:
        raise HTTPException(
            status_code=404,
            detail=f"model {body.model!r} is not served by this process",
        )
    if (
        body.max_tokens is not None
        and body.max_completion_tokens is not None
        and body.max_tokens != body.max_completion_tokens
    ):
        raise HTTPException(
            status_code=400,
            detail="max_tokens and max_completion_tokens disagree",
        )
    max_tokens = (
        body.max_completion_tokens
        if body.max_completion_tokens is not None
        else body.max_tokens
    )
    if max_tokens is None:
        max_tokens = settings.default_max_tokens
    block_length = body.block_length or settings.default_block_length
    if block_length > max_tokens:
        raise HTTPException(
            status_code=400,
            detail="block_length must be less than or equal to max_tokens",
        )
    if body.temperature != 0:
        raise HTTPException(status_code=400, detail="only temperature=0 is supported")
    if body.top_p != 1:
        raise HTTPException(status_code=400, detail="only top_p=1 is supported")
    if body.n != 1:
        raise HTTPException(status_code=400, detail="only n=1 is supported")
    return max_tokens, block_length


def _base_chunk(
    *, request_id: str, created: int, model: str, choices: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }


def create_app(
    settings: ServerSettings,
    engine_factory: Callable[[], Any],
) -> FastAPI:
    service = LocalChatService(
        engine_factory,
        max_total_tokens=settings.max_total_tokens,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.start()
        app.state.service = service
        app.state.generations = set()
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title="OLED-MoE Local Server", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                str(exc.detail), code="invalid_request", error_type="invalid_request_error"
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", ())[1:])
        message = first.get("msg", "invalid request")
        if location:
            message = f"{location}: {message}"
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                message, code="validation_error", error_type="invalid_request_error"
            ),
        )

    @app.get("/health")
    @app.get("/healthz")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "ready": service.ready,
            "model": settings.served_model_name,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        max_tokens, block_length = _resolve_request(body, settings)
        messages = [message.model_dump() for message in body.messages]
        try:
            prepared = await service.prepare(messages, max_tokens=max_tokens)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model_name = settings.served_model_name

        if not body.stream:
            try:
                result = await service.generate(
                    prepared,
                    max_tokens=max_tokens,
                    block_length=block_length,
                )
            except Exception as exc:
                logger.exception("chat generation %s failed", request_id)
                raise HTTPException(status_code=500, detail="generation failed") from exc
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": result.finish_reason,
                    }
                ],
                "usage": _usage(result),
            }

        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_block(block: TextBlock) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ("block", block))

        async def produce() -> None:
            try:
                result = await service.generate(
                    prepared,
                    max_tokens=max_tokens,
                    block_length=block_length,
                    on_block=on_block,
                )
            except Exception:
                logger.exception("streaming chat generation %s failed", request_id)
                await queue.put(("error", None))
            else:
                await queue.put(("result", result))

        task = asyncio.create_task(produce())
        app.state.generations.add(task)
        task.add_done_callback(app.state.generations.discard)

        async def events():
            role = _base_chunk(
                request_id=request_id,
                created=created,
                model=model_name,
                choices=[
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            )
            yield _sse(role)
            while True:
                kind, payload = await queue.get()
                if kind == "block":
                    block = payload
                    assert isinstance(block, TextBlock)
                    chunk = _base_chunk(
                        request_id=request_id,
                        created=created,
                        model=model_name,
                        choices=[
                            {
                                "index": 0,
                                "delta": {"content": block.delta},
                                "finish_reason": None,
                            }
                        ],
                    )
                    chunk["oled_block_index"] = block.index
                    chunk["oled_block_tokens"] = block.token_count
                    chunk["oled_text"] = block.text
                    yield _sse(chunk)
                    continue
                if kind == "error":
                    yield _sse(
                        _error_payload(
                            "generation failed",
                            code="generation_failed",
                            error_type="server_error",
                        )
                    )
                    yield "data: [DONE]\n\n"
                    break
                result = payload
                assert isinstance(result, GenerationResult)
                final = _base_chunk(
                    request_id=request_id,
                    created=created,
                    model=model_name,
                    choices=[
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": result.finish_reason,
                        }
                    ],
                )
                yield _sse(final)
                if body.stream_options and body.stream_options.include_usage:
                    usage_chunk = _base_chunk(
                        request_id=request_id,
                        created=created,
                        model=model_name,
                        choices=[],
                    )
                    usage_chunk["usage"] = _usage(result)
                    yield _sse(usage_chunk)
                yield "data: [DONE]\n\n"
                break

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return app
