from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from .expert_store import PinnedExpertStore

_EXPERT_KEY = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


class LLaDA2CheckpointLoader:
    """Stream dense tensors to the model and expert tensors to the pinned store."""

    def __init__(self, *, dtype: torch.dtype = torch.bfloat16) -> None:
        self.dtype = dtype

    def load(self, model: torch.nn.Module, model_dir: str | Path, store: PinnedExpertStore) -> None:
        model_dir = Path(model_dir)
        index_path = model_dir / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        shard_files = sorted(set(index["weight_map"].values()))
        parameters = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        loaded_dense: set[str] = set()

        for shard_name in shard_files:
            shard_path = model_dir / shard_name
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118 - safetensors handle is not a dict
                    value = handle.get_tensor(key)
                    match = _EXPERT_KEY.match(key)
                    if match is not None:
                        store.layer(int(match.group("layer"))).put(
                            int(match.group("expert")),
                            match.group("projection"),
                            value,
                        )
                        continue
                    target = parameters.get(key)
                    if target is None:
                        target = buffers.get(key)
                    if target is None:
                        continue
                    target_dtype = torch.float32 if key.endswith("expert_bias") else self.dtype
                    target.data = value.to(device="cpu", dtype=target_dtype)
                    loaded_dense.add(key)

        store.validate_complete()
        missing_dense = [
            name for name in parameters if ".mlp.experts." not in name and name not in loaded_dense
        ]
        allowed_missing = {"h2e.embedding.weight"}
        unexpected = [name for name in missing_dense if name not in allowed_missing]
        if unexpected:
            raise RuntimeError(f"missing dense checkpoint tensors: {unexpected[:12]}")
