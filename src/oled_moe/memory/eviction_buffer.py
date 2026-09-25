from __future__ import annotations

import torch


class DeviceEvictionBuffer:
    """Request-local GPU spill space for experts displaced by miss execution."""

    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self.packed: torch.Tensor | None = None
        self.capacity = 0
        self.packed_width = 0
        self.expert_to_slot: dict[int, int] = {}

    @property
    def allocated_bytes(self) -> int:
        if self.packed is None:
            return 0
        return self.packed.numel() * self.packed.element_size()

    def begin(
        self,
        *,
        capacity: int,
        packed_width: int,
        stream: torch.cuda.Stream,
    ) -> None:
        self.expert_to_slot.clear()
        if capacity <= 0:
            return
        if (
            self.packed is None
            or self.capacity < capacity
            or self.packed_width != packed_width
        ):
            # Growth is rare and happens during warmup. Synchronizing the shared
            # copy stream prevents replacing storage that an earlier layer still reads.
            stream.synchronize()
            self.packed = torch.empty(
                (capacity, packed_width),
                device=self.device,
                dtype=self.dtype,
            )
            self.capacity = capacity
            self.packed_width = packed_width

    def capture(self, expert_id: int, source: torch.Tensor) -> None:
        expert_id = int(expert_id)
        if expert_id in self.expert_to_slot:
            return
        if self.packed is None:
            raise RuntimeError("eviction buffer has not been allocated")
        slot = len(self.expert_to_slot)
        if slot >= self.capacity:
            raise RuntimeError("eviction buffer capacity was underestimated")
        self.packed[slot].copy_(source, non_blocking=True)
        self.expert_to_slot[expert_id] = slot

    def source(self, expert_id: int) -> torch.Tensor:
        if self.packed is None or expert_id not in self.expert_to_slot:
            raise KeyError(f"expert {expert_id} was not captured in the eviction buffer")
        return self.packed[self.expert_to_slot[expert_id]]
