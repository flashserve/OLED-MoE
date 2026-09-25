from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from oled_moe.runtime import MoERuntime


def create_confidence_tracking_decoder(
    *,
    runtime: MoERuntime,
    temperature: float,
    threshold: float,
    mask_id: int,
    eos_id: int,
    use_float64: bool = False,
) -> Any:
    """Create a dInfer threshold decoder that exposes its existing confidence."""

    from dinfer.decoding.parallel_strategy import (
        ThresholdParallelDecoder,
        broadcast_if_needed,
    )
    from dinfer.decoding.utils import add_gumbel_noise

    @torch.compile(dynamic=True)
    def decode_with_confidence(
        logits: torch.Tensor,
        current_tokens: torch.Tensor,
        mask: torch.Tensor,
        iteration_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        probability_dtype = torch.float64 if use_float64 else torch.float32
        probabilities = F.softmax(logits.to(probability_dtype), dim=-1)
        confidence = probabilities.gather(
            dim=-1,
            index=predicted_tokens.unsqueeze(-1),
        ).squeeze(-1)
        valid_mask = mask & (predicted_tokens != mask_id)
        predicted_tokens = torch.where(valid_mask, predicted_tokens, current_tokens)
        masked_confidence = torch.where(
            valid_mask,
            confidence,
            torch.full_like(confidence, -torch.inf),
        )
        actual_threshold = (
            masked_confidence.max(dim=1)[0] - 1e-5
        ).clamp(-1000, iteration_threshold).unsqueeze(-1)
        transfer_mask = (masked_confidence >= actual_threshold) & mask
        observed_confidence = torch.where(
            valid_mask,
            confidence,
            torch.zeros_like(confidence),
        )
        return predicted_tokens, transfer_mask, observed_confidence.float()

    class ConfidenceTrackingThresholdDecoder(ThresholdParallelDecoder):
        def __init__(self) -> None:
            super().__init__(
                temperature=temperature,
                threshold=threshold,
                mask_id=mask_id,
                eos_id=eos_id,
                use_float64=use_float64,
            )
            self._block_confidence: torch.Tensor | None = None

        def block_init(self, block_x: torch.Tensor, block_id: int) -> None:
            del block_id
            self._block_confidence = torch.ones_like(block_x, dtype=torch.float32)
            runtime.set_previous_confidence(self._block_confidence)

        def decode(
            self,
            logits: torch.Tensor,
            block_start: int,
            block_end: int,
            x: Any,
            iter_threshold: float | None = None,
        ) -> None:
            if iter_threshold is None:
                iter_threshold = self.threshold
            current_tokens = x[:, block_start:block_end]
            mask = current_tokens == self.mask_id
            if mask.shape[1] != logits.shape[1]:
                raise ValueError("decoder mask and logits lengths differ")
            predicted, transfer_mask, observed_confidence = decode_with_confidence(
                logits,
                current_tokens,
                mask,
                iter_threshold,
            )
            transfer_mask = transfer_mask & mask
            x[:, block_start:block_end] = torch.where(
                transfer_mask,
                predicted,
                current_tokens,
            )
            broadcast_if_needed(x.data)

            if (
                self._block_confidence is None
                or self._block_confidence.shape != observed_confidence.shape
            ):
                self._block_confidence = torch.ones_like(
                    observed_confidence,
                    dtype=torch.float32,
                )
            self._block_confidence = torch.where(
                mask,
                observed_confidence,
                self._block_confidence,
            )
            runtime.set_previous_confidence(self._block_confidence)

    return ConfidenceTrackingThresholdDecoder()
