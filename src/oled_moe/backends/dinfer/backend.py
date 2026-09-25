from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from transformers import AutoConfig, AutoTokenizer

from oled_moe.config import OLEDConfig
from oled_moe.runtime import MoERuntime

from .block_stream import completed_block_callback
from .confidence_decoder import create_confidence_tracking_decoder
from .environment import configure_dinfer_path
from .model_bridge import build_llada2_model


@dataclass
class DInferSession:
    model: torch.nn.Module
    tokenizer: object
    generator: object
    runtime: MoERuntime
    device: torch.device
    _generate_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        gen_length: int,
        block_length: int,
        on_block: Callable[[torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        with self._generate_lock:
            self.runtime.begin_request(block_length=block_length)
            with completed_block_callback(
                self.generator,
                prompt_length=int(input_ids.shape[-1]),
                gen_length=gen_length,
                on_block=on_block,
            ):
                return self.generator.generate(
                    input_ids,
                    gen_length=gen_length,
                    block_length=block_length,
                )

    def close(self) -> None:
        try:
            self.runtime.close()
        finally:
            _destroy_distributed()


def _destroy_distributed() -> None:
    try:
        from vllm import distributed

        distributed.destroy_model_parallel()
        distributed.destroy_distributed_environment()
    except (RuntimeError, AttributeError):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class DInferBackend:
    """Single-GPU LLaDA2 adapter for the upstream dInfer checkout."""

    def __init__(self, *, dinfer_path: str | None = None) -> None:
        self.dinfer_path = configure_dinfer_path(dinfer_path)

    def create_session(
        self,
        *,
        model_path: str,
        oled_config: OLEDConfig,
        device: str | torch.device = "cuda:0",
        threshold: float = 0.9,
        cache_kind: str = "prefix",
    ) -> DInferSession:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("the first OLED milestone supports CUDA only")
        if device.index not in (None, 0):
            raise ValueError("use CUDA_VISIBLE_DEVICES to select a physical GPU and pass cuda:0")
        torch.cuda.set_device(device)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", os.environ.get("OLED_MASTER_PORT", "45711"))

        from dinfer import (
            BlockDiffusionLLM,
            BlockIteratorFactory,
            KVCacheFactory,
            ThresholdParallelDecoder,
        )
        from vllm import distributed
        from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

        if torch.distributed.is_initialized():
            raise RuntimeError("a distributed process group is already initialized")
        distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
        distributed.initialize_model_parallel(1, backend="nccl")

        try:
            runtime = MoERuntime(oled_config, dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            parallel = ParallelConfig(enable_expert_parallel=True)
            with set_current_vllm_config(VllmConfig(parallel_config=parallel)):
                model = build_llada2_model(
                    model_config=model_config,
                    model_path=model_path,
                    runtime=runtime,
                    device=device,
                    dtype=torch.bfloat16,
                )

            decoder_kwargs = {
                "temperature": 0,
                "threshold": threshold,
                "mask_id": 156895,
                "eos_id": 156892,
            }
            if oled_config.cache.policy == "priority":
                decoder = create_confidence_tracking_decoder(
                    runtime=runtime,
                    **decoder_kwargs,
                )
            else:
                decoder = ThresholdParallelDecoder(**decoder_kwargs)
            cache_factory = KVCacheFactory(cache_kind, is_bd_model=True)
            generator = BlockDiffusionLLM(
                model,
                decoder,
                BlockIteratorFactory(start_block_align=True, use_block_diffusion=True),
                cache_factory=cache_factory,
                early_stop=True,
            )
            return DInferSession(model, tokenizer, generator, runtime, device)
        except BaseException:
            _destroy_distributed()
            raise
