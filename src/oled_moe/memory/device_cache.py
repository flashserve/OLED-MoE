from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from .device_pool import DeviceExpertPool
from .eviction_buffer import DeviceEvictionBuffer
from .expert_store import LayerExpertStore


@dataclass(frozen=True)
class TransferTicket:
    expert_ids: tuple[int, ...]
    load_count: int
    loaded_bytes: int
    copy_submissions: int
    ready_event: torch.cuda.Event
    dma_interval: tuple[torch.cuda.Event, torch.cuda.Event] | None = None


class LayerDeviceCache:
    """Preallocated GPU slots with LRU replacement and asynchronous H2D loads."""

    def __init__(
        self,
        *,
        layer_id: int,
        store: LayerExpertStore,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype,
        use_copy_stream: bool,
        coalesce_expert_weights: bool = True,
        copy_stream: torch.cuda.Stream | None = None,
        pool: DeviceExpertPool | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.layer_id = layer_id
        self.store = store
        self.capacity = min(capacity, store.shape.num_experts)
        self.device = device
        self.dtype = dtype
        self.coalesce_expert_weights = coalesce_expert_weights
        self.pool = pool
        packed_width = 3 * store.shape.intermediate_size * store.shape.hidden_size
        if pool is not None:
            if (
                pool.device != device
                or pool.dtype != dtype
                or pool.hidden_size != store.shape.hidden_size
                or pool.intermediate_size != store.shape.intermediate_size
            ):
                raise ValueError("global expert pool is incompatible with the layer cache")
            self.packed = pool.packed
        else:
            self.packed = torch.empty(
                (self.capacity, packed_width),
                device=device,
                dtype=dtype,
            )
        self.expert_to_slot: dict[int, int] = {}
        self.slot_to_expert: dict[int, int] = {}
        self.last_used: dict[int, int] = {}
        self._clock = 0
        self.copy_stream = None
        if use_copy_stream and device.type == "cuda":
            self.copy_stream = copy_stream or torch.cuda.Stream(device=device)

    @property
    def w13(self) -> torch.Tensor:
        h = self.store.shape.hidden_size
        i = self.store.shape.intermediate_size
        slots = self.pool.total_slots if self.pool is not None else self.capacity
        return self.packed.as_strided(
            size=(slots, 2 * i, h),
            stride=(3 * i * h, h, 1),
        )

    @property
    def w2(self) -> torch.Tensor:
        h = self.store.shape.hidden_size
        i = self.store.shape.intermediate_size
        slots = self.pool.total_slots if self.pool is not None else self.capacity
        return self.packed.as_strided(
            size=(slots, h, i),
            stride=(3 * i * h, i, 1),
            storage_offset=2 * i * h,
        )

    @property
    def resident_experts(self) -> frozenset[int]:
        return frozenset(self.expert_to_slot)

    def slot(self, expert_id: int) -> int:
        return self.expert_to_slot[expert_id]

    def available_slots(self, protected_experts: Iterable[int]) -> int:
        protected = {int(expert) for expert in protected_experts}
        missing = protected.difference(self.expert_to_slot)
        if missing:
            raise KeyError(f"cannot protect non-resident experts: {sorted(missing)}")
        return self.capacity - len(protected)

    def _choose_slot(
        self,
        protected: set[int],
    ) -> tuple[int, int | None, torch.cuda.Event | None]:
        if len(self.expert_to_slot) < self.capacity:
            if self.pool is not None:
                slot_id, reusable_after = self.pool.acquire(self.layer_id)
                return slot_id, None, reusable_after
            for slot_id in range(self.capacity):
                if slot_id not in self.slot_to_expert:
                    return slot_id, None, None
        candidates = [expert for expert in self.expert_to_slot if expert not in protected]
        if not candidates:
            raise RuntimeError(
                f"layer {self.layer_id} cache has no replaceable slot; "
                f"capacity={self.capacity}, protected={len(protected)}"
            )
        victim = min(
            candidates,
            key=lambda expert: self.last_used.get(expert, -1),
        )
        return self.expert_to_slot[victim], victim, None

    def resize(
        self,
        capacity: int,
        *,
        keep_experts: Iterable[int],
        reusable_after: torch.cuda.Event | None = None,
    ) -> int:
        """Change the logical quota and release surplus global slots."""

        capacity = min(int(capacity), self.store.shape.num_experts)
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        keep = {int(expert) for expert in keep_experts}
        if not keep.issubset(self.expert_to_slot):
            raise KeyError("cannot keep a non-resident expert while resizing")
        if len(keep) > capacity:
            raise ValueError("kept experts exceed the resized cache capacity")
        if len(self.expert_to_slot) > capacity and len(keep) < capacity:
            extras = sorted(
                set(self.expert_to_slot).difference(keep),
                key=lambda expert: self.last_used.get(expert, -1),
                reverse=True,
            )
            keep.update(extras[: capacity - len(keep)])
        victims = set(self.expert_to_slot).difference(keep)
        for expert_id in victims:
            slot_id = self.expert_to_slot.pop(expert_id)
            self.slot_to_expert.pop(slot_id)
            self.last_used.pop(expert_id, None)
            if self.pool is not None:
                self.pool.release(
                    layer_id=self.layer_id,
                    slot_id=slot_id,
                    reusable_after=reusable_after,
                )
        self.capacity = capacity
        return len(victims)

    def begin_transfer(
        self,
        expert_ids: Iterable[int],
        *,
        protected_experts: Iterable[int] = (),
        slots_reusable_after: torch.cuda.Event | None = None,
        eviction_buffer: DeviceEvictionBuffer | None = None,
        capture_experts: Iterable[int] | None = None,
        profile: bool = False,
    ) -> TransferTicket:
        requested = tuple(dict.fromkeys(int(expert) for expert in expert_ids))
        if not requested:
            raise ValueError("begin_transfer requires at least one expert")
        protected = {int(expert) for expert in protected_experts}
        capture = None if capture_experts is None else {int(expert) for expert in capture_experts}
        if set(requested).intersection(protected):
            raise ValueError("requested and protected expert sets must be disjoint")
        misses = [expert for expert in requested if expert not in self.expert_to_slot]
        if len(misses) != len(requested):
            raise ValueError("begin_transfer accepts cache misses only")
        if len(requested) > self.available_slots(protected):
            raise ValueError(
                f"requested {len(requested)} experts with only "
                f"{self.available_slots(protected)} replaceable slots"
            )

        assignments: list[tuple[int, int, int | None, torch.cuda.Event | None]] = []
        replacement_protected = protected.union(requested)
        for expert_id in misses:
            slot_id, old_expert, pool_reusable_after = self._choose_slot(replacement_protected)
            if old_expert is not None:
                self.expert_to_slot.pop(old_expert, None)
                self.last_used.pop(old_expert, None)
            self.slot_to_expert[slot_id] = expert_id
            self.expert_to_slot[expert_id] = slot_id
            assignments.append((expert_id, slot_id, old_expert, pool_reusable_after))

        compute_stream = torch.cuda.current_stream(self.device)
        transfer_stream = self.copy_stream or compute_stream
        dma_interval = None
        if profile:
            dma_interval = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )

        with torch.cuda.stream(transfer_stream):
            if slots_reusable_after is not None:
                transfer_stream.wait_event(slots_reusable_after)
            for _, _, _, pool_reusable_after in assignments:
                if pool_reusable_after is not None:
                    transfer_stream.wait_event(pool_reusable_after)
            if dma_interval is not None:
                dma_interval[0].record(transfer_stream)
            for expert_id, slot_id, old_expert, _ in assignments:
                if (
                    eviction_buffer is not None
                    and old_expert is not None
                    and (capture is None or old_expert in capture)
                ):
                    eviction_buffer.capture(old_expert, self.packed[slot_id])
                if self.coalesce_expert_weights:
                    self.packed[slot_id].copy_(
                        self.store.packed[expert_id],
                        non_blocking=self.copy_stream is not None,
                    )
                else:
                    # Diagnostic control path for measuring one packed copy versus
                    # the previous two projection-specific H2D submissions.
                    self.w13[slot_id].copy_(
                        self.store.w13[expert_id],
                        non_blocking=self.copy_stream is not None,
                    )
                    self.w2[slot_id].copy_(
                        self.store.w2[expert_id],
                        non_blocking=self.copy_stream is not None,
                    )
            if dma_interval is not None:
                dma_interval[1].record(transfer_stream)
            ready_event = torch.cuda.Event()
            ready_event.record(transfer_stream)

        self.touch(requested)
        return TransferTicket(
            expert_ids=requested,
            load_count=len(assignments),
            loaded_bytes=len(assignments) * self.store.expert_bytes,
            copy_submissions=len(assignments) * (1 if self.coalesce_expert_weights else 2),
            ready_event=ready_event,
            dma_interval=dma_interval,
        )

    def begin_restore(
        self,
        expert_ids: Iterable[int],
        *,
        eviction_buffer: DeviceEvictionBuffer,
        protected_experts: Iterable[int] = (),
        slots_reusable_after: torch.cuda.Event | None = None,
        profile: bool = False,
    ) -> TransferTicket:
        requested = tuple(dict.fromkeys(int(expert) for expert in expert_ids))
        if not requested:
            raise ValueError("begin_restore requires at least one expert")
        protected = {int(expert) for expert in protected_experts}
        if set(requested).intersection(protected):
            raise ValueError("requested and protected expert sets must be disjoint")
        if any(expert in self.expert_to_slot for expert in requested):
            raise ValueError("begin_restore accepts non-resident experts only")
        if len(requested) > self.available_slots(protected):
            raise ValueError("not enough replaceable cache slots for restore")

        assignments: list[tuple[int, int, torch.cuda.Event | None]] = []
        replacement_protected = protected.union(requested)
        for expert_id in requested:
            slot_id, old_expert, pool_reusable_after = self._choose_slot(replacement_protected)
            if old_expert is not None:
                self.expert_to_slot.pop(old_expert, None)
                self.last_used.pop(old_expert, None)
            self.slot_to_expert[slot_id] = expert_id
            self.expert_to_slot[expert_id] = slot_id
            assignments.append((expert_id, slot_id, pool_reusable_after))

        compute_stream = torch.cuda.current_stream(self.device)
        transfer_stream = self.copy_stream or compute_stream
        dma_interval = None
        if profile:
            dma_interval = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        with torch.cuda.stream(transfer_stream):
            if slots_reusable_after is not None:
                transfer_stream.wait_event(slots_reusable_after)
            for _, _, pool_reusable_after in assignments:
                if pool_reusable_after is not None:
                    transfer_stream.wait_event(pool_reusable_after)
            if dma_interval is not None:
                dma_interval[0].record(transfer_stream)
            for expert_id, slot_id, _ in assignments:
                self.packed[slot_id].copy_(
                    eviction_buffer.source(expert_id),
                    non_blocking=self.copy_stream is not None,
                )
            if dma_interval is not None:
                dma_interval[1].record(transfer_stream)
            ready_event = torch.cuda.Event()
            ready_event.record(transfer_stream)

        self.touch(requested)
        return TransferTicket(
            expert_ids=requested,
            load_count=len(assignments),
            loaded_bytes=len(assignments) * self.store.expert_bytes,
            copy_submissions=len(assignments),
            ready_event=ready_event,
            dma_interval=dma_interval,
        )

    def wait_for_transfer(
        self,
        ticket: TransferTicket,
        *,
        profile: bool = False,
    ) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
        compute_stream = torch.cuda.current_stream(self.device)
        wait_interval = None
        if profile:
            wait_interval = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            wait_interval[0].record(compute_stream)
        compute_stream.wait_event(ticket.ready_event)
        if wait_interval is not None:
            wait_interval[1].record(compute_stream)
        return wait_interval

    def touch(self, expert_ids: Iterable[int]) -> None:
        for expert_id in expert_ids:
            self._clock += 1
            self.last_used[int(expert_id)] = self._clock
