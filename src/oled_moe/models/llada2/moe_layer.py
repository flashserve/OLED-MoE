from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from oled_moe.runtime import MoERuntime

_LAYER_PATTERN = re.compile(r"layers\.(\d+)")


class LLaDA2OffloadedExperts(nn.Module):
    """FusedMoE-compatible facade backed by the OLED GPU-only runtime."""

    def __init__(
        self,
        *,
        runtime: MoERuntime,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        custom_routing_function: Callable[..., Any] | None = None,
        prefix: str = "",
        renormalize: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        match = _LAYER_PATTERN.search(prefix)
        if match is None:
            raise ValueError(f"cannot determine layer id from dInfer prefix: {prefix!r}")
        self.layer_id = int(match.group(1))
        self.runtime = runtime
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.renormalize = bool(renormalize)
        self.custom_routing_function = custom_routing_function
        self.expert_map = torch.arange(self.num_experts, dtype=torch.int32)
        runtime.register_layer(
            layer_id=self.layer_id,
            num_experts=self.num_experts,
            hidden_size=int(hidden_size),
            intermediate_size=int(intermediate_size),
        )

    def forward_impl(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        if self.custom_routing_function is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            gate_weights, expert_ids = torch.topk(scores, self.top_k, dim=-1)
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)
        else:
            routed = self.custom_routing_function(
                hidden_states,
                router_logits,
                self.top_k,
                self.renormalize,
            )
            gate_weights, expert_ids = routed[:2]
        return self.runtime.execute(
            layer_id=self.layer_id,
            hidden_states=hidden_states,
            expert_ids=expert_ids,
            gate_weights=gate_weights,
        )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        return self.forward_impl(hidden_states, router_logits)
