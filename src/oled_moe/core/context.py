from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RequestContext:
    """Request-local state reserved for confidence-aware policies."""

    previous_confidence: torch.Tensor | None = None
    block_index: int = 0
