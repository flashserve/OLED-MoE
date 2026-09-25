from __future__ import annotations

import torch
import torch.nn.functional as F

from oled_moe.backends.dinfer.confidence_decoder import (
    create_confidence_tracking_decoder,
)


class _RuntimeRecorder:
    def __init__(self) -> None:
        self.confidence: torch.Tensor | None = None

    def set_previous_confidence(self, confidence: torch.Tensor) -> None:
        self.confidence = confidence.detach().clone()


def test_tracking_decoder_matches_upstream_and_reuses_decoder_confidence() -> None:
    from dinfer.decoding.parallel_strategy import ThresholdParallelDecoder

    torch.manual_seed(31)
    mask_id = 9
    logits = torch.randn(1, 4, 10)
    logits[..., mask_id] = -10
    baseline_tokens = torch.full((1, 4), mask_id)
    tracked_tokens = baseline_tokens.clone()
    baseline = ThresholdParallelDecoder(
        temperature=0,
        threshold=0.9,
        mask_id=mask_id,
        eos_id=8,
    )
    recorder = _RuntimeRecorder()
    tracked = create_confidence_tracking_decoder(
        runtime=recorder,  # type: ignore[arg-type]
        temperature=0,
        threshold=0.9,
        mask_id=mask_id,
        eos_id=8,
    )
    tracked.block_init(tracked_tokens, 0)

    baseline.decode(logits, 0, 4, baseline_tokens)
    tracked.decode(logits, 0, 4, tracked_tokens)

    torch.testing.assert_close(tracked_tokens, baseline_tokens)
    expected = F.softmax(logits.float(), dim=-1).max(dim=-1).values
    if recorder.confidence is None:
        raise AssertionError("decoder did not publish confidence")
    torch.testing.assert_close(recorder.confidence, expected)
