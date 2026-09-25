from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch


class GenerationSession(Protocol):
    model: torch.nn.Module

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        gen_length: int,
        block_length: int,
        on_block: Callable[[torch.Tensor], None] | None = None,
    ) -> torch.Tensor: ...

    def close(self) -> None: ...
