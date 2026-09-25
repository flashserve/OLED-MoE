from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExpertShape:
    num_experts: int
    hidden_size: int
    intermediate_size: int


class LayerExpertStore:
    """Expert-major CPU bank with one contiguous, addressable slice per expert."""

    def __init__(
        self,
        *,
        layer_id: int,
        shape: ExpertShape,
        dtype: torch.dtype,
        pin_memory: bool,
    ) -> None:
        self.layer_id = layer_id
        self.shape = shape
        self.dtype = dtype
        self.pin_memory = pin_memory
        self._packed: torch.Tensor | None = None
        self._loaded_parts = torch.zeros((shape.num_experts, 3), dtype=torch.bool)

    def _allocate(self) -> None:
        if self._packed is not None:
            return
        e = self.shape.num_experts
        h = self.shape.hidden_size
        i = self.shape.intermediate_size
        self._packed = torch.empty(
            (e, 3 * i * h),
            dtype=self.dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        )

    @property
    def packed(self) -> torch.Tensor:
        """Packed ``[w13 | w2]`` storage; each expert row is H2D contiguous."""

        if self._packed is None:
            raise RuntimeError(f"expert bank for layer {self.layer_id} has not been loaded")
        return self._packed

    @property
    def w13(self) -> torch.Tensor:
        e = self.shape.num_experts
        h = self.shape.hidden_size
        i = self.shape.intermediate_size
        return self.packed.as_strided(
            size=(e, 2 * i, h),
            stride=(3 * i * h, h, 1),
        )

    @property
    def w2(self) -> torch.Tensor:
        e = self.shape.num_experts
        h = self.shape.hidden_size
        i = self.shape.intermediate_size
        return self.packed.as_strided(
            size=(e, h, i),
            stride=(3 * i * h, i, 1),
            storage_offset=2 * i * h,
        )

    @property
    def expert_bytes(self) -> int:
        h = self.shape.hidden_size
        i = self.shape.intermediate_size
        elements = 2 * i * h + h * i
        return elements * torch.empty((), dtype=self.dtype).element_size()

    @property
    def allocated_bytes(self) -> int:
        if self._packed is None:
            return 0
        return self._packed.numel() * self._packed.element_size()

    def put(self, expert_id: int, projection: str, value: torch.Tensor) -> None:
        self._allocate()
        if not 0 <= expert_id < self.shape.num_experts:
            raise IndexError(f"expert_id {expert_id} is outside layer {self.layer_id}")
        value = value.detach().to(device="cpu", dtype=self.dtype)
        i = self.shape.intermediate_size
        if projection == "gate_proj":
            expected = self.w13[expert_id, :i]
            part = 0
        elif projection == "up_proj":
            expected = self.w13[expert_id, i:]
            part = 1
        elif projection == "down_proj":
            expected = self.w2[expert_id]
            part = 2
        else:
            raise ValueError(f"unknown expert projection: {projection}")
        if tuple(value.shape) != tuple(expected.shape):
            raise ValueError(
                f"shape mismatch for layer={self.layer_id} expert={expert_id} "
                f"projection={projection}: expected {tuple(expected.shape)}, got {tuple(value.shape)}"
            )
        expected.copy_(value)
        self._loaded_parts[expert_id, part] = True

    def validate_complete(self) -> None:
        missing = (~self._loaded_parts.all(dim=1)).nonzero(as_tuple=False).flatten().tolist()
        if missing:
            preview = missing[:8]
            raise RuntimeError(
                f"layer {self.layer_id} has {len(missing)} incomplete experts; first={preview}"
            )
        if self.pin_memory and not self.packed.is_pinned():
            raise RuntimeError(f"layer {self.layer_id} expert bank is not pinned")


class PinnedExpertStore:
    """All routed expert weights, indexed by layer and expert."""

    def __init__(self, *, dtype: torch.dtype = torch.bfloat16, pin_memory: bool = True) -> None:
        self.dtype = dtype
        self.pin_memory = pin_memory
        self._layers: dict[int, LayerExpertStore] = {}

    def register_layer(
        self,
        *,
        layer_id: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> LayerExpertStore:
        shape = ExpertShape(num_experts, hidden_size, intermediate_size)
        existing = self._layers.get(layer_id)
        if existing is not None:
            if existing.shape != shape:
                raise ValueError(f"layer {layer_id} was registered with a different shape")
            return existing
        layer = LayerExpertStore(
            layer_id=layer_id,
            shape=shape,
            dtype=self.dtype,
            pin_memory=self.pin_memory,
        )
        self._layers[layer_id] = layer
        return layer

    def layer(self, layer_id: int) -> LayerExpertStore:
        try:
            return self._layers[layer_id]
        except KeyError as exc:
            raise KeyError(f"unregistered expert layer {layer_id}") from exc

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._layers))

    @property
    def allocated_bytes(self) -> int:
        return sum(layer.allocated_bytes for layer in self._layers.values())

    def validate_complete(self) -> None:
        for layer in self._layers.values():
            layer.validate_complete()
