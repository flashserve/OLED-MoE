from __future__ import annotations

from collections.abc import Callable

import torch

from oled_moe.backends.dinfer import DInferBackend, DInferSession
from oled_moe.config import OLEDConfig


class Engine:
    """Small public facade; backend-specific details stay behind the session."""

    def __init__(self, session: DInferSession) -> None:
        self.session = session

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        config: OLEDConfig | None = None,
        device: str = "cuda:0",
        threshold: float = 0.9,
        dinfer_path: str | None = None,
    ) -> Engine:
        backend = DInferBackend(dinfer_path=dinfer_path)
        session = backend.create_session(
            model_path=model_path,
            oled_config=config or OLEDConfig(),
            device=device,
            threshold=threshold,
        )
        return cls(session)

    @property
    def tokenizer(self) -> object:
        return self.session.tokenizer

    @property
    def runtime(self):
        return self.session.runtime

    def generate_ids(
        self,
        input_ids: torch.Tensor,
        *,
        gen_length: int,
        block_length: int,
        on_block: Callable[[torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        return self.session.generate(
            input_ids.to(self.session.device),
            gen_length=gen_length,
            block_length=block_length,
            on_block=on_block,
        )

    def close(self) -> None:
        self.session.close()
