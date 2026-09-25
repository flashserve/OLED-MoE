from __future__ import annotations

import torch


class DeviceExpertPool:
    """One preallocated GPU expert-slot bank shared by compatible MoE layers."""

    def __init__(
        self,
        *,
        total_slots: int,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if total_slots <= 0:
            raise ValueError("device expert pool must contain at least one slot")
        self.total_slots = total_slots
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.device = device
        self.dtype = dtype
        self.packed_width = 3 * intermediate_size * hidden_size
        self.packed = torch.empty(
            (total_slots, self.packed_width),
            device=device,
            dtype=dtype,
        )
        self.slot_owner: dict[int, int] = {}
        self._free_slots = list(range(total_slots - 1, -1, -1))
        self._release_events: dict[int, torch.cuda.Event] = {}

    @property
    def allocated_bytes(self) -> int:
        return self.packed.numel() * self.packed.element_size()

    @property
    def free_slots(self) -> int:
        return len(self._free_slots)

    @property
    def w13(self) -> torch.Tensor:
        return self.packed.as_strided(
            size=(self.total_slots, 2 * self.intermediate_size, self.hidden_size),
            stride=(self.packed_width, self.hidden_size, 1),
        )

    @property
    def w2(self) -> torch.Tensor:
        return self.packed.as_strided(
            size=(self.total_slots, self.hidden_size, self.intermediate_size),
            stride=(self.packed_width, self.intermediate_size, 1),
            storage_offset=2 * self.intermediate_size * self.hidden_size,
        )

    def acquire(self, layer_id: int) -> tuple[int, torch.cuda.Event | None]:
        if not self._free_slots:
            raise RuntimeError("global GPU expert pool has no free slot")
        slot_id = self._free_slots.pop()
        if slot_id in self.slot_owner:
            raise AssertionError("free global expert slot still has an owner")
        self.slot_owner[slot_id] = layer_id
        return slot_id, self._release_events.pop(slot_id, None)

    def release(
        self,
        *,
        layer_id: int,
        slot_id: int,
        reusable_after: torch.cuda.Event | None = None,
    ) -> None:
        if self.slot_owner.get(slot_id) != layer_id:
            raise ValueError(
                f"layer {layer_id} cannot release global slot {slot_id} "
                f"owned by {self.slot_owner.get(slot_id)}"
            )
        self.slot_owner.pop(slot_id)
        self._free_slots.append(slot_id)
        if reusable_after is None:
            self._release_events.pop(slot_id, None)
        else:
            self._release_events[slot_id] = reusable_after
