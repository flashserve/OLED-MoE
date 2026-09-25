from __future__ import annotations

from collections.abc import Iterable

import torch

from .device_cache import TransferTicket
from .expert_store import LayerExpertStore


class DeviceTransientExpertBuffer:
    """Reusable GPU scratch slots for experts that must not enter the cache."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        use_copy_stream: bool,
        coalesce_expert_weights: bool = True,
        copy_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.coalesce_expert_weights = coalesce_expert_weights
        self.copy_stream = None
        if use_copy_stream and device.type == "cuda":
            self.copy_stream = copy_stream or torch.cuda.Stream(device=device)
        self.packed: torch.Tensor | None = None
        self.capacity = 0
        self.packed_width = 0
        self.store: LayerExpertStore | None = None
        self.expert_to_slot: dict[int, int] = {}
        self._last_use_event: torch.cuda.Event | None = None

    @property
    def allocated_bytes(self) -> int:
        if self.packed is None:
            return 0
        return self.packed.numel() * self.packed.element_size()

    @property
    def w13(self) -> torch.Tensor:
        if self.packed is None or self.store is None:
            raise RuntimeError("transient expert buffer is empty")
        hidden = self.store.shape.hidden_size
        intermediate = self.store.shape.intermediate_size
        return self.packed.as_strided(
            size=(self.capacity, 2 * intermediate, hidden),
            stride=(3 * intermediate * hidden, hidden, 1),
        )

    @property
    def w2(self) -> torch.Tensor:
        if self.packed is None or self.store is None:
            raise RuntimeError("transient expert buffer is empty")
        hidden = self.store.shape.hidden_size
        intermediate = self.store.shape.intermediate_size
        return self.packed.as_strided(
            size=(self.capacity, hidden, intermediate),
            stride=(3 * intermediate * hidden, intermediate, 1),
            storage_offset=2 * intermediate * hidden,
        )

    def slot(self, expert_id: int) -> int:
        return self.expert_to_slot[expert_id]

    def begin_transfer(
        self,
        expert_ids: Iterable[int],
        *,
        store: LayerExpertStore,
        profile: bool = False,
    ) -> TransferTicket:
        requested = tuple(dict.fromkeys(int(expert) for expert in expert_ids))
        if not requested:
            raise ValueError("transient transfer requires at least one expert")
        packed_width = 3 * store.shape.intermediate_size * store.shape.hidden_size
        needs_growth = (
            self.packed is None
            or self.capacity < len(requested)
            or self.packed_width != packed_width
        )
        if needs_growth:
            if self._last_use_event is not None:
                # Growth is expected during warmup. Do not release storage still
                # consumed by an earlier layer on the compute stream.
                self._last_use_event.synchronize()
            self.packed = torch.empty(
                (len(requested), packed_width),
                device=self.device,
                dtype=self.dtype,
            )
            self.capacity = len(requested)
            self.packed_width = packed_width
            self._last_use_event = None
        self.store = store
        self.expert_to_slot = {expert_id: slot_id for slot_id, expert_id in enumerate(requested)}

        compute_stream = torch.cuda.current_stream(self.device)
        transfer_stream = self.copy_stream or compute_stream
        dma_interval = None
        if profile:
            dma_interval = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        if self.packed is None:
            raise AssertionError("transient expert storage was not allocated")
        with torch.cuda.stream(transfer_stream):
            if self._last_use_event is not None:
                transfer_stream.wait_event(self._last_use_event)
            if dma_interval is not None:
                dma_interval[0].record(transfer_stream)
            for slot_id, expert_id in enumerate(requested):
                if self.coalesce_expert_weights:
                    self.packed[slot_id].copy_(
                        store.packed[expert_id],
                        non_blocking=self.copy_stream is not None,
                    )
                else:
                    self.w13[slot_id].copy_(
                        store.w13[expert_id],
                        non_blocking=self.copy_stream is not None,
                    )
                    self.w2[slot_id].copy_(
                        store.w2[expert_id],
                        non_blocking=self.copy_stream is not None,
                    )
            if dma_interval is not None:
                dma_interval[1].record(transfer_stream)
            ready_event = torch.cuda.Event()
            ready_event.record(transfer_stream)

        copy_multiplier = 1 if self.coalesce_expert_weights else 2
        return TransferTicket(
            expert_ids=requested,
            load_count=len(requested),
            loaded_bytes=len(requested) * store.expert_bytes,
            copy_submissions=len(requested) * copy_multiplier,
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

    def mark_used(self, done_event: torch.cuda.Event) -> None:
        self._last_use_event = done_event
