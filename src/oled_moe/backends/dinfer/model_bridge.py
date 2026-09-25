from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial

import torch

from oled_moe.memory import LLaDA2CheckpointLoader
from oled_moe.models import LLaDA2OffloadedExperts
from oled_moe.runtime import MoERuntime


def _track_model_forward(
    runtime: MoERuntime,
    _module: torch.nn.Module,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    model_input = kwargs.get("input_ids")
    if model_input is None:
        model_input = kwargs.get("inputs_embeds")
    if model_input is None and args:
        model_input = args[0]
    if not isinstance(model_input, torch.Tensor) or model_input.ndim < 2:
        return
    token_span = int(model_input.shape[-2]) if model_input.ndim >= 3 else int(model_input.shape[-1])
    runtime.begin_model_forward(token_span=token_span)


@contextmanager
def _offloaded_moe_factory(runtime: MoERuntime) -> Iterator[object]:
    from dinfer.model import modeling_llada2_moe

    original = modeling_llada2_moe.FusedMoE
    modeling_llada2_moe.FusedMoE = partial(LLaDA2OffloadedExperts, runtime=runtime)
    try:
        yield modeling_llada2_moe
    finally:
        modeling_llada2_moe.FusedMoE = original


def build_llada2_model(
    *,
    model_config: object,
    model_path: str,
    runtime: MoERuntime,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    with _offloaded_moe_factory(runtime) as modeling:
        model = modeling.LLaDA2MoeModelLM(config=model_config)
    loader = LLaDA2CheckpointLoader(dtype=dtype)
    loader.load(model, model_path, runtime.store)
    model.to(device=device)
    runtime.prepare(device)
    model.register_forward_pre_hook(
        partial(_track_model_forward, runtime),
        with_kwargs=True,
    )
    model.eval()
    return model
