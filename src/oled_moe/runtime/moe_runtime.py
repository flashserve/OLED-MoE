from __future__ import annotations

import concurrent.futures
import json
import math
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from oled_moe.config import OLEDConfig
from oled_moe.core import ExpertKey
from oled_moe.executor import CPUExecutionResult, CPUExpertExecutor, CPUPreparedExecution
from oled_moe.memory import (
    DeviceEvictionBuffer,
    DeviceExpertPool,
    DeviceTransientExpertBuffer,
    LayerDeviceCache,
    PinnedExpertStore,
    TransferTicket,
)
from oled_moe.observability import LayerMetrics
from oled_moe.placement import (
    LayerCapacityController,
    StaticPlacementPolicy,
    build_priority_route_packet,
    observe_and_rank_retention_candidates,
    select_retained_experts,
)
from oled_moe.scheduler import CooperativeMissScheduler, GPUOnlyMissScheduler


@dataclass(frozen=True)
class _PendingCudaTiming:
    layer_id: int
    metric: str
    start: torch.cuda.Event
    end: torch.cuda.Event


@dataclass(frozen=True)
class _ChunkExecution:
    done_event: torch.cuda.Event
    timing_interval: tuple[torch.cuda.Event, torch.cuda.Event] | None


class MoERuntime:
    """Model-independent expert runtime with GPU-only and static placement modes."""

    def __init__(self, config: OLEDConfig, *, dtype: torch.dtype = torch.bfloat16) -> None:
        self.config = config
        self.dtype = dtype
        self.store = PinnedExpertStore(dtype=dtype, pin_memory=config.pin_memory)
        self.scheduler = (
            CooperativeMissScheduler(
                cpu_miss_ratio=config.placement.cpu_miss_ratio,
                transfer_high_priority=config.placement.transfer_high_priority,
                cpu_min_dispatch_experts=(
                    config.placement.cpu_min_dispatch_experts
                ),
            )
            if config.placement.mode == "cooperative"
            else GPUOnlyMissScheduler()
        )
        self.layer_capacity_controller = LayerCapacityController(config.layer_capacity)
        self.static_placement = (
            StaticPlacementPolicy(config.placement)
            if config.placement.mode in {"cpu_only", "static_gpu", "static_hybrid"}
            else None
        )
        self.cpu_executor = (
            None
            if config.placement.mode in {"gpu_only", "static_gpu"}
            else CPUExpertExecutor(
                num_threads=config.cpu.num_threads,
                strategy=config.cpu.strategy,
                pointwise_mode=config.cpu.pointwise_mode,
            )
        )
        self._caches: dict[int, LayerDeviceCache] = {}
        self._device_pools: dict[tuple[int, int, int], DeviceExpertPool] = {}
        self._metrics: dict[int, LayerMetrics] = {}
        self._pending_timings: list[_PendingCudaTiming] = []
        self._retention_tickets: dict[int, TransferTicket] = {}
        self._route_frequency: dict[int, np.ndarray] = {}
        self._route_gate_mass: dict[int, np.ndarray] = {}
        self._priority_steps: dict[int, int] = {}
        self._last_retention_scores: dict[int, dict[int, float]] = {}
        self._layer_priority_opportunity_ema: dict[int, float] = {}
        self._layer_priority_observations: dict[int, int] = {}
        self._layer_priority_enabled: dict[int, bool] = {}
        self._copy_streams: dict[int, torch.cuda.Stream] = {}
        self._eviction_buffers: dict[int, DeviceEvictionBuffer] = {}
        self._transient_buffers: dict[int, DeviceTransientExpertBuffer] = {}
        self._previous_confidence: torch.Tensor | None = None
        self._applied_layer_capacity_updates = 0
        self._layer_capacity_released_slots = 0
        self._layer_capacity_observe_ns = 0
        self._layer_capacity_update_ns = 0
        self._layer_capacity_apply_ns = 0
        self._cache_trace_path = os.environ.get("OLED_CACHE_TRACE_PATH")
        self._cache_trace: list[dict[str, object]] = []
        self._request_id = -1
        self._request_block_length: int | None = None
        self._observe_layer_capacity_this_forward = True
        self._layer_capacity_skipped_forwards = 0
        self._static_prepared_layers: set[int] = set()
        self._cpu_dispatch_pool = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="oled-cpu-dispatch",
            )
            if config.placement.mode == "cooperative"
            else None
        )

    def register_layer(
        self,
        *,
        layer_id: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.store.register_layer(
            layer_id=layer_id,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.layer_capacity_controller.register_layer(
            layer_id=layer_id,
            num_experts=num_experts,
            initial_capacity=self.config.cache.capacity_per_layer,
        )
        self._metrics.setdefault(layer_id, LayerMetrics())

    def _device_pool(
        self,
        layer_id: int,
        device: torch.device,
    ) -> DeviceExpertPool:
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        store = self.store.layer(layer_id)
        device_index = device.index if device.index is not None else 0
        key = (
            device_index,
            store.shape.hidden_size,
            store.shape.intermediate_size,
        )
        pool = self._device_pools.get(key)
        if pool is None:
            compatible_layers = [
                self.store.layer(candidate)
                for candidate in self.store.layer_ids
                if (
                    self.store.layer(candidate).shape.hidden_size == store.shape.hidden_size
                    and self.store.layer(candidate).shape.intermediate_size
                    == store.shape.intermediate_size
                )
            ]
            total_slots = sum(
                min(self.config.cache.capacity_per_layer, layer.shape.num_experts)
                for layer in compatible_layers
            )
            pool = DeviceExpertPool(
                total_slots=total_slots,
                hidden_size=store.shape.hidden_size,
                intermediate_size=store.shape.intermediate_size,
                device=device,
                dtype=self.dtype,
            )
            self._device_pools[key] = pool
        return pool

    def _cache(self, layer_id: int, device: torch.device) -> LayerDeviceCache:
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        cache = self._caches.get(layer_id)
        if cache is None:
            copy_stream = None
            if self.config.transfer.use_copy_stream and device.type == "cuda":
                device_index = (
                    device.index if device.index is not None else torch.cuda.current_device()
                )
                copy_stream = self._copy_streams.get(device_index)
                if copy_stream is None:
                    copy_stream = torch.cuda.Stream(device=device)
                    self._copy_streams[device_index] = copy_stream
            cache = LayerDeviceCache(
                layer_id=layer_id,
                store=self.store.layer(layer_id),
                capacity=self.config.cache.capacity_per_layer,
                device=device,
                dtype=self.dtype,
                use_copy_stream=self.config.transfer.use_copy_stream,
                coalesce_expert_weights=self.config.transfer.coalesce_expert_weights,
                copy_stream=copy_stream,
                pool=self._device_pool(layer_id, device),
            )
            self._caches[layer_id] = cache
        elif cache.device != device:
            raise RuntimeError(f"layer {layer_id} cache already belongs to {cache.device}")
        return cache

    def _transient_buffer(self, device: torch.device) -> DeviceTransientExpertBuffer:
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        device_index = device.index if device.index is not None else 0
        buffer = self._transient_buffers.get(device_index)
        if buffer is None:
            buffer = DeviceTransientExpertBuffer(
                device=device,
                dtype=self.dtype,
                use_copy_stream=self.config.transfer.use_copy_stream,
                coalesce_expert_weights=self.config.transfer.coalesce_expert_weights,
            )
            self._transient_buffers[device_index] = buffer
        return buffer

    def _placement(self, layer_id: int):
        if self.static_placement is None:
            raise RuntimeError("static placement is unavailable in dynamic execution modes")
        layer = self.store.layer(layer_id)
        return self.static_placement.plan(
            layer_id=layer_id,
            num_experts=layer.shape.num_experts,
        )

    def _prepare_static_layer(self, layer_id: int, device: torch.device) -> None:
        if layer_id in self._static_prepared_layers:
            return
        placement = self._placement(layer_id)
        if not placement.gpu_experts:
            self._static_prepared_layers.add(layer_id)
            return
        if len(placement.gpu_experts) > self.config.cache.capacity_per_layer:
            raise ValueError(
                f"layer {layer_id} static placement needs {len(placement.gpu_experts)} "
                f"GPU slots but cache capacity is {self.config.cache.capacity_per_layer}"
            )
        cache = self._cache(layer_id, device)
        ticket = cache.begin_transfer(placement.gpu_experts)
        cache.wait_for_transfer(ticket)
        self._static_prepared_layers.add(layer_id)

    def prepare(self, device: torch.device) -> None:
        """Materialize immutable static GPU placements before generation begins."""

        if self.config.placement.mode not in {"static_gpu", "static_hybrid"}:
            return
        for layer_id in self.store.layer_ids:
            self._prepare_static_layer(layer_id, device)
        torch.cuda.synchronize(device)

    def begin_request(self, *, block_length: int | None = None) -> None:
        """Reset request-local state while preserving the warmed expert cache."""

        if block_length is not None and block_length <= 0:
            raise ValueError("block_length must be positive")
        self._request_id += 1
        self._request_block_length = block_length
        self._observe_layer_capacity_this_forward = True
        self._previous_confidence = None

    def begin_model_forward(self, *, token_span: int) -> None:
        """Mark whether this model forward represents exactly one decoding block."""

        if token_span <= 0:
            raise ValueError("token_span must be positive")
        eligible = self._request_block_length is None or token_span == self._request_block_length
        self._observe_layer_capacity_this_forward = eligible
        if self.config.layer_capacity.enabled and not eligible:
            self._layer_capacity_skipped_forwards += 1

    def set_previous_confidence(self, confidence: torch.Tensor) -> None:
        """Publish decoder confidence for the next transformer forward."""

        self._previous_confidence = confidence.detach()

    def reconfigure_cache_capacity(self, capacity_per_layer: int) -> None:
        """Start a new benchmark scenario without reloading pinned model weights."""

        if capacity_per_layer <= 0:
            raise ValueError("capacity_per_layer must be positive")
        if self.config.placement.mode in {"cpu_only", "static_gpu", "static_hybrid"}:
            raise RuntimeError("cache capacity reconfiguration requires dynamic placement")
        if (
            self._caches
            or self._retention_tickets
            or self._pending_timings
            or self._transient_buffers
        ):
            torch.cuda.synchronize()
        self._pending_timings.clear()
        self._retention_tickets.clear()
        self._caches.clear()
        self._device_pools.clear()
        self._eviction_buffers.clear()
        self._transient_buffers.clear()
        self._route_frequency.clear()
        self._route_gate_mass.clear()
        self._priority_steps.clear()
        self._last_retention_scores.clear()
        self._layer_priority_opportunity_ema.clear()
        self._layer_priority_observations.clear()
        self._layer_priority_enabled.clear()
        self._previous_confidence = None
        self._request_block_length = None
        self._observe_layer_capacity_this_forward = True
        self._layer_capacity_skipped_forwards = 0
        self._applied_layer_capacity_updates = 0
        self._layer_capacity_released_slots = 0
        self._layer_capacity_observe_ns = 0
        self._layer_capacity_update_ns = 0
        self._layer_capacity_apply_ns = 0
        self.layer_capacity_controller.reset(initial_capacity=capacity_per_layer)
        self.config = replace(
            self.config,
            cache=replace(
                self.config.cache,
                capacity_per_layer=capacity_per_layer,
            ),
        )
        for layer_id in self._metrics:
            self._metrics[layer_id] = LayerMetrics()
        torch.cuda.empty_cache()

    def _maybe_apply_layer_capacities(
        self,
        *,
        layer_id: int,
        device: torch.device,
    ) -> None:
        if not self.config.layer_capacity.apply:
            return
        if not self.store.layer_ids or layer_id != self.store.layer_ids[0]:
            return
        controller = self.layer_capacity_controller
        if controller.update_count <= self._applied_layer_capacity_updates:
            return
        apply_started = time.perf_counter_ns() if self.config.profiling.enabled else None
        capacities = controller.recommended_capacities()
        if set(capacities) != set(self._caches):
            raise RuntimeError("layer-capacity application requires every registered cache")
        pools = {id(cache.pool): cache.pool for cache in self._caches.values()}
        if None in pools.values() or len(pools) != 1:
            raise RuntimeError(
                "online layer-capacity application currently requires one "
                "compatible global expert pool"
            )
        pool = next(iter(pools.values()))
        if pool is None or sum(capacities.values()) != pool.total_slots:
            raise AssertionError("layer capacities do not match the global pool")

        reusable_after = torch.cuda.Event()
        reusable_after.record(torch.cuda.current_stream(device))
        released = 0
        donors = sorted(
            (
                candidate
                for candidate, cache in self._caches.items()
                if capacities[candidate] < cache.capacity
            ),
            key=lambda candidate: capacities[candidate] - self._caches[candidate].capacity,
        )
        for candidate in donors:
            cache = self._caches[candidate]
            scores = self._last_retention_scores.get(candidate, {})
            keep = select_retained_experts(
                cache.resident_experts,
                priorities=scores,
                recency=cache.last_used,
                capacity=capacities[candidate],
            )
            released += cache.resize(
                capacities[candidate],
                keep_experts=keep,
                reusable_after=reusable_after,
            )
        for candidate, cache in sorted(self._caches.items()):
            if capacities[candidate] > cache.capacity:
                cache.resize(
                    capacities[candidate],
                    keep_experts=cache.resident_experts,
                )
        if sum(cache.capacity for cache in self._caches.values()) != pool.total_slots:
            raise AssertionError("applied layer quotas changed the global budget")
        controller.mark_applied(capacities)
        self._applied_layer_capacity_updates = controller.update_count
        self._layer_capacity_released_slots += released
        if apply_started is not None:
            self._layer_capacity_apply_ns += time.perf_counter_ns() - apply_started

    def _add_timing(
        self,
        layer_id: int,
        metric: str,
        interval: tuple[torch.cuda.Event, torch.cuda.Event] | None,
    ) -> None:
        if interval is not None:
            self._pending_timings.append(
                _PendingCudaTiming(layer_id, metric, interval[0], interval[1])
            )

    def _drain_timings(self) -> None:
        if not self._pending_timings:
            return
        torch.cuda.synchronize()
        for timing in self._pending_timings:
            metrics = self._metrics[timing.layer_id]
            elapsed_ms = timing.start.elapsed_time(timing.end)
            setattr(metrics, timing.metric, getattr(metrics, timing.metric) + elapsed_ms)
        self._pending_timings.clear()

    @torch.no_grad()
    def execute(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        gate_weights: torch.Tensor,
    ) -> torch.Tensor:
        execute_started = time.perf_counter_ns()
        if hidden_states.device.type != "cuda":
            raise RuntimeError("OLED-MoE requires CUDA-resident hidden states")
        if expert_ids.shape != gate_weights.shape:
            raise ValueError("expert_ids and gate_weights must have identical shapes")
        if expert_ids.ndim != 2 or hidden_states.ndim != 2:
            raise ValueError("expected hidden_states [N,H] and routes [N,K]")
        if expert_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError("route token count does not match hidden_states")

        if self.config.placement.mode == "static_gpu":
            return self._execute_static_gpu(
                layer_id=layer_id,
                hidden_states=hidden_states,
                expert_ids=expert_ids,
                gate_weights=gate_weights,
                execute_started=execute_started,
            )
        if self.config.placement.mode in {"cpu_only", "static_hybrid"}:
            return self._execute_static(
                layer_id=layer_id,
                hidden_states=hidden_states,
                expert_ids=expert_ids,
                gate_weights=gate_weights,
                execute_started=execute_started,
            )

        profile = self.config.profiling.enabled
        device = hidden_states.device
        capacity_eligible = self._observe_layer_capacity_this_forward
        if capacity_eligible:
            self._maybe_apply_layer_capacities(layer_id=layer_id, device=device)
        cache = self._cache(layer_id, device)
        metrics = self._metrics[layer_id]
        self._await_retention(layer_id, cache, metrics, profile=profile)

        route_started = time.perf_counter_ns()
        resident_before = cache.resident_experts
        flat_experts = expert_ids.reshape(-1)
        num_experts = self.store.layer(layer_id).shape.num_experts
        cache_policy = self.config.cache.policy
        policy_interval = self._priority_update_interval(num_experts)
        policy_step = self._priority_steps.get(layer_id, 0)
        self._priority_steps[layer_id] = policy_step + 1
        policy_scheduled = policy_step % policy_interval == 0
        policy_update = policy_scheduled and self._priority_enabled_for_layer(layer_id)
        if policy_scheduled:
            metrics.priority_scheduled_calls += 1
            if policy_update:
                metrics.priority_eligible_calls += 1
            else:
                metrics.priority_bypassed_calls += 1

        # All dynamic cache policies pay the same route-observation cost. They
        # differ only in the score used to select the retained expert set.
        observation_started = time.perf_counter_ns()
        observation_timing = None
        if profile:
            observation_timing = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            observation_timing[0].record(torch.cuda.current_stream(device))
        route_packet = build_priority_route_packet(
            expert_ids=expert_ids,
            gate_weights=gate_weights,
            previous_confidence=self._previous_confidence,
            confidence_power=self.config.cache.confidence_power,
            gate_power=self.config.cache.gate_power,
            include_raw_gates=(
                self.config.placement.mode == "cooperative"
                and self.config.cpu.compact_io
                and self.scheduler.cpu_miss_ratio > 0.0
            ),
        )
        if observation_timing is not None:
            observation_timing[1].record(torch.cuda.current_stream(device))
            self._add_timing(layer_id, "cuda_priority_ms", observation_timing)
        route_packet_host = route_packet.cpu().numpy()
        route_frequency = self._route_frequency.get(layer_id)
        if route_frequency is None:
            route_frequency = np.zeros(num_experts, dtype=np.int64)
            self._route_frequency[layer_id] = route_frequency
        route_gate_mass = self._route_gate_mass.get(layer_id)
        if route_gate_mass is None:
            route_gate_mass = np.zeros(num_experts, dtype=np.float64)
            self._route_gate_mass[layer_id] = route_gate_mass
        ranking = observe_and_rank_retention_candidates(
            route_packet=route_packet_host,
            token_count=expert_ids.shape[0],
            num_experts=num_experts,
            resident_experts=resident_before,
            previous_recency=cache.last_used,
            route_frequency=route_frequency,
            route_gate_mass=route_gate_mass,
            policy=cache_policy,
        )
        metrics.route_accounting_calls += 1
        metrics.route_accounted_pairs += int(flat_experts.numel())
        metrics.priority_updates += 1
        unique_experts = list(ranking.active_experts)
        route_counts = ranking.route_counts
        priority_by_expert = ranking.current_priorities
        selection_recency = ranking.recency

        if self._cache_trace_path:
            self._cache_trace.append(
                {
                    "request_id": self._request_id,
                    "layer_id": layer_id,
                    "cache_policy": cache_policy,
                    "active_experts": unique_experts,
                    "priorities": [priority_by_expert[expert] for expert in unique_experts],
                }
            )
        if profile:
            metrics.host_priority_ms += (time.perf_counter_ns() - observation_started) / 1e6

        hit_ids = [expert for expert in unique_experts if expert in resident_before]
        miss_ids = [expert for expert in unique_experts if expert not in resident_before]
        expert_priorities = {
            ExpertKey(layer_id, expert): priority_by_expert[expert] for expert in miss_ids
        }
        plan = self.scheduler.plan(
            hits=(ExpertKey(layer_id, expert) for expert in hit_ids),
            misses=(ExpertKey(layer_id, expert) for expert in miss_ids),
            priorities=expert_priorities,
        )
        if self.config.placement.mode == "gpu_only" and plan.cpu_experts:
            raise AssertionError("GPU-only scheduler produced CPU work")
        transfer_ids = [expert.expert_id for expert in plan.transfer_experts]
        cpu_ids = tuple(expert.expert_id for expert in plan.cpu_experts)
        if set(hit_ids).union(transfer_ids, cpu_ids) != set(unique_experts):
            raise AssertionError("dynamic scheduler did not partition every active expert")

        gpu_candidate_set = set(resident_before).union(transfer_ids)
        ranked_gpu_candidates = tuple(
            expert for expert in ranking.ranked_experts if expert in gpu_candidate_set
        )
        retention_priority_by_expert = ranking.selected_scores
        self._last_retention_scores[layer_id] = dict(retention_priority_by_expert)

        def retention_order(expert: int) -> tuple[float, int, int]:
            return (
                retention_priority_by_expert.get(expert, 0.0),
                selection_recency.get(expert, -1),
                -expert,
            )

        if policy_update:
            retained_target = ranked_gpu_candidates[: cache.capacity]
            metrics.retention_updates += 1
        else:
            # Between common policy-update points, keep the fixed cache stable;
            # active misses use the transient buffer and cannot displace it.
            retained_target = tuple(sorted(resident_before, key=retention_order, reverse=True))
        retained_set = set(retained_target)
        hit_ids.sort(key=retention_order, reverse=True)
        transfer_ids.sort(
            key=lambda expert: (
                expert in retained_set,
                *retention_order(expert),
            )
        )
        eviction_buffer = None
        if profile:
            metrics.host_route_plan_ms += (time.perf_counter_ns() - route_started) / 1e6

        stats_started = time.perf_counter_ns()
        pair_hits = sum(route_counts[expert] for expert in hit_ids)
        total_pairs = int(flat_experts.numel())
        self._record_cache_quality(
            metrics,
            active_experts=len(unique_experts),
            hit_experts=len(hit_ids),
            capacity=cache.capacity,
            occupied_slots=len(resident_before),
        )
        capacity_started = time.perf_counter_ns() if profile and capacity_eligible else None
        capacity_updated = False
        if capacity_eligible:
            capacity_updated = self.layer_capacity_controller.observe(
                layer_id=layer_id,
                active_experts=len(unique_experts),
                hit_experts=len(hit_ids),
                capacity=cache.capacity,
            )
        if capacity_started is not None:
            capacity_elapsed = time.perf_counter_ns() - capacity_started
            self._layer_capacity_observe_ns += capacity_elapsed
            if capacity_updated:
                self._layer_capacity_update_ns += capacity_elapsed
        self._observe_layer_priority_opportunity(
            layer_id=layer_id,
            active_experts=len(unique_experts),
            hit_experts=len(hit_ids),
            capacity=cache.capacity,
        )
        if profile:
            metrics.host_cache_stats_ms += (time.perf_counter_ns() - stats_started) / 1e6

        dispatch_started = time.perf_counter_ns()
        gpu_route_output = torch.zeros(
            (total_pairs, hidden_states.shape[-1]),
            device=device,
            dtype=hidden_states.dtype,
        )
        if retained_target:
            cache.touch(
                sorted(
                    hit_ids,
                    key=retention_order,
                )
            )
        else:
            cache.touch(hit_ids)
        remaining_misses = list(transfer_ids)

        if retained_target:
            resident_retained = sorted(
                resident_before.intersection(retained_set),
                key=retention_order,
            )
            persistent_misses = [expert for expert in transfer_ids if expert in retained_set]
            transient_misses = [expert for expert in transfer_ids if expert not in retained_set]

            transient_buffer = None
            transient_ticket = None
            if transient_misses:
                transient_buffer = self._transient_buffer(device)
                transient_ticket = self._begin_transient_transfer(
                    layer_id=layer_id,
                    buffer=transient_buffer,
                    metrics=metrics,
                    experts=transient_misses,
                    store=cache.store,
                    profile=profile,
                )

            protected_before_hit = set(hit_ids).union(resident_retained)
            preload_count = min(
                len(persistent_misses),
                cache.available_slots(protected_before_hit),
            )
            preloaded_persistent = persistent_misses[:preload_count]
            deferred_persistent = persistent_misses[preload_count:]
            persistent_tickets: list[TransferTicket] = []
            if preloaded_persistent:
                persistent_tickets.append(
                    self._begin_transfer(
                        layer_id=layer_id,
                        cache=cache,
                        metrics=metrics,
                        experts=preloaded_persistent,
                        protected=protected_before_hit,
                        slots_reusable_after=None,
                        eviction_buffer=None,
                        capture_experts=(),
                        profile=profile,
                    )
                )

            cpu_prepared: CPUPreparedExecution | None = None
            cpu_future: concurrent.futures.Future[CPUExecutionResult] | None = None
            if cpu_ids:
                if self.cpu_executor is None or self._cpu_dispatch_pool is None:
                    raise AssertionError("cooperative mode did not initialize CPU execution")
                if self.config.cpu.compact_io:
                    cpu_prepared = self.cpu_executor.prepare_compact(
                        store=self.store.layer(layer_id),
                        hidden_states=hidden_states,
                        expert_ids=expert_ids,
                        route_packet_host=route_packet_host,
                        active_experts=cpu_ids,
                        route_counts=route_counts,
                        profile=profile,
                        workspace_scope=layer_id,
                    )
                else:
                    cpu_prepared = self.cpu_executor.prepare(
                        store=self.store.layer(layer_id),
                        hidden_states=hidden_states,
                        expert_ids=expert_ids,
                        gate_weights=gate_weights,
                        active_experts=cpu_ids,
                        route_counts=route_counts,
                        profile=profile,
                    )
                cpu_future = self._cpu_dispatch_pool.submit(
                    self.cpu_executor.execute_prepared,
                    cpu_prepared,
                )
                metrics.cpu_dispatch_submissions += 1

            hit_execution = self._execute_chunk(
                cache=cache,
                metrics=metrics,
                hidden_states=hidden_states,
                expert_ids=expert_ids,
                gate_weights=gate_weights,
                chunk=hit_ids,
                route_counts=route_counts,
                output=gpu_route_output,
                profile=profile,
            )
            hit_done = hit_execution.done_event if hit_execution is not None else None
            if hit_execution is not None:
                self._add_timing(
                    layer_id,
                    "cuda_moe_compute_ms",
                    hit_execution.timing_interval,
                )

            if deferred_persistent:
                persistent_tickets.append(
                    self._begin_transfer(
                        layer_id=layer_id,
                        cache=cache,
                        metrics=metrics,
                        experts=deferred_persistent,
                        protected=resident_retained + preloaded_persistent,
                        slots_reusable_after=hit_done,
                        eviction_buffer=None,
                        capture_experts=(),
                        profile=profile,
                    )
                )
            for persistent_ticket in persistent_tickets:
                wait_interval = cache.wait_for_transfer(
                    persistent_ticket,
                    profile=profile,
                )
                self._add_timing(
                    layer_id,
                    "cuda_transfer_wait_ms",
                    wait_interval,
                )
            if transient_ticket is not None:
                if transient_buffer is None:
                    raise AssertionError("transient transfer lost its buffer")
                wait_interval = transient_buffer.wait_for_transfer(
                    transient_ticket,
                    profile=profile,
                )
                self._add_timing(
                    layer_id,
                    "cuda_transfer_wait_ms",
                    wait_interval,
                )

            miss_execution = None
            if persistent_misses and transient_misses:
                if transient_buffer is None:
                    raise AssertionError("mixed miss execution lost its buffer")
                miss_execution = self._execute_chunk(
                    cache=cache,
                    fallback_cache=transient_buffer,
                    metrics=metrics,
                    hidden_states=hidden_states,
                    expert_ids=expert_ids,
                    gate_weights=gate_weights,
                    chunk=transfer_ids,
                    route_counts=route_counts,
                    output=gpu_route_output,
                    profile=profile,
                )
            elif persistent_misses:
                miss_execution = self._execute_chunk(
                    cache=cache,
                    metrics=metrics,
                    hidden_states=hidden_states,
                    expert_ids=expert_ids,
                    gate_weights=gate_weights,
                    chunk=persistent_misses,
                    route_counts=route_counts,
                    output=gpu_route_output,
                    profile=profile,
                )
            elif transient_misses:
                if transient_buffer is None:
                    raise AssertionError("transient miss execution lost its buffer")
                miss_execution = self._execute_chunk(
                    cache=transient_buffer,
                    metrics=metrics,
                    hidden_states=hidden_states,
                    expert_ids=expert_ids,
                    gate_weights=gate_weights,
                    chunk=transient_misses,
                    route_counts=route_counts,
                    output=gpu_route_output,
                    profile=profile,
                )
            if miss_execution is not None:
                if transient_buffer is not None:
                    transient_buffer.mark_used(miss_execution.done_event)
                self._add_timing(
                    layer_id,
                    "cuda_moe_compute_ms",
                    miss_execution.timing_interval,
                )

            cache.touch(
                sorted(
                    retained_target,
                    key=retention_order,
                )
            )
            if cache.resident_experts != retained_set:
                raise AssertionError(
                    "priority dispatch did not leave the selected experts resident"
                )
        else:
            current_ticket: TransferTicket | None = None
            replaceable = cache.available_slots(hit_ids)
            first_count = self._pipeline_chunk_size(
                remaining=len(remaining_misses),
                replaceable=replaceable,
                capacity=cache.capacity,
            )
            if first_count:
                first_chunk = remaining_misses[:first_count]
                remaining_misses = remaining_misses[first_count:]
                current_ticket = self._begin_transfer(
                    layer_id=layer_id,
                    cache=cache,
                    metrics=metrics,
                    experts=first_chunk,
                    protected=hit_ids,
                    slots_reusable_after=None,
                    eviction_buffer=eviction_buffer,
                    capture_experts=retained_set,
                    profile=profile,
                )

            cpu_prepared: CPUPreparedExecution | None = None
            cpu_future: concurrent.futures.Future[CPUExecutionResult] | None = None
            if cpu_ids:
                if self.cpu_executor is None or self._cpu_dispatch_pool is None:
                    raise AssertionError("cooperative mode did not initialize CPU execution")
                if self.config.cpu.compact_io:
                    cpu_prepared = self.cpu_executor.prepare_compact(
                        store=self.store.layer(layer_id),
                        hidden_states=hidden_states,
                        expert_ids=expert_ids,
                        route_packet_host=route_packet_host,
                        active_experts=cpu_ids,
                        route_counts=route_counts,
                        profile=profile,
                        workspace_scope=layer_id,
                    )
                else:
                    cpu_prepared = self.cpu_executor.prepare(
                        store=self.store.layer(layer_id),
                        hidden_states=hidden_states,
                        expert_ids=expert_ids,
                        gate_weights=gate_weights,
                        active_experts=cpu_ids,
                        route_counts=route_counts,
                        profile=profile,
                    )
                cpu_future = self._cpu_dispatch_pool.submit(
                    self.cpu_executor.execute_prepared,
                    cpu_prepared,
                )
                metrics.cpu_dispatch_submissions += 1

            current_ticket_waited = False
            if current_ticket is not None and not self.config.transfer.overlap_compute:
                wait_interval = cache.wait_for_transfer(current_ticket, profile=profile)
                self._add_timing(layer_id, "cuda_transfer_wait_ms", wait_interval)
                current_ticket_waited = True

            hit_execution = self._execute_chunk(
                cache=cache,
                metrics=metrics,
                hidden_states=hidden_states,
                expert_ids=expert_ids,
                gate_weights=gate_weights,
                chunk=hit_ids,
                route_counts=route_counts,
                output=gpu_route_output,
                profile=profile,
            )
            hit_done = hit_execution.done_event if hit_execution is not None else None
            if hit_execution is not None:
                self._add_timing(
                    layer_id,
                    "cuda_moe_compute_ms",
                    hit_execution.timing_interval,
                )

            if current_ticket is None and remaining_misses:
                first_count = self._pipeline_chunk_size(
                    remaining=len(remaining_misses),
                    replaceable=cache.capacity,
                    capacity=cache.capacity,
                )
                first_chunk = remaining_misses[:first_count]
                remaining_misses = remaining_misses[first_count:]
                current_ticket = self._begin_transfer(
                    layer_id=layer_id,
                    cache=cache,
                    metrics=metrics,
                    experts=first_chunk,
                    protected=(),
                    slots_reusable_after=hit_done,
                    eviction_buffer=eviction_buffer,
                    capture_experts=retained_set,
                    profile=profile,
                )

            slots_reusable_after = hit_done
            while current_ticket is not None:
                if not current_ticket_waited:
                    wait_interval = cache.wait_for_transfer(current_ticket, profile=profile)
                    self._add_timing(layer_id, "cuda_transfer_wait_ms", wait_interval)
                current_ticket_waited = False
                current_chunk = list(current_ticket.expert_ids)
                current_execution = self._execute_chunk(
                    cache=cache,
                    metrics=metrics,
                    hidden_states=hidden_states,
                    expert_ids=expert_ids,
                    gate_weights=gate_weights,
                    chunk=current_chunk,
                    route_counts=route_counts,
                    output=gpu_route_output,
                    profile=profile,
                )
                if current_execution is None:
                    raise AssertionError("a transferred expert chunk had no routed tokens")
                self._add_timing(
                    layer_id,
                    "cuda_moe_compute_ms",
                    current_execution.timing_interval,
                )

                next_ticket = None
                if remaining_misses:
                    replaceable = cache.available_slots(current_chunk)
                    next_count = self._pipeline_chunk_size(
                        remaining=len(remaining_misses),
                        replaceable=replaceable,
                        capacity=cache.capacity,
                    )
                    if next_count <= 0:
                        raise RuntimeError("double-buffer pipeline made no forward progress")
                    next_chunk = remaining_misses[:next_count]
                    remaining_misses = remaining_misses[next_count:]
                    next_ticket = self._begin_transfer(
                        layer_id=layer_id,
                        cache=cache,
                        metrics=metrics,
                        experts=next_chunk,
                        protected=current_chunk,
                        slots_reusable_after=(
                            slots_reusable_after
                            if self.config.transfer.overlap_compute
                            else current_execution.done_event
                        ),
                        eviction_buffer=eviction_buffer,
                        capture_experts=retained_set,
                        profile=profile,
                    )
                slots_reusable_after = current_execution.done_event
                current_ticket = next_ticket

            if retained_target:
                self._schedule_policy_retention(
                    layer_id=layer_id,
                    cache=cache,
                    metrics=metrics,
                    retained=retained_target,
                    eviction_buffer=eviction_buffer,
                    slots_reusable_after=slots_reusable_after,
                    profile=profile,
                )

        output = gpu_route_output.view(
            expert_ids.shape[0],
            expert_ids.shape[1],
            hidden_states.shape[-1],
        ).sum(dim=1)
        cpu_pairs = sum(route_counts[expert] for expert in cpu_ids)
        if cpu_future is not None:
            cpu_wait_started = time.perf_counter_ns()
            cpu_host_result = cpu_future.result()
            if profile:
                metrics.host_cpu_wait_ms += (time.perf_counter_ns() - cpu_wait_started) / 1e6
            if cpu_prepared is None or self.cpu_executor is None:
                raise AssertionError("CPU future completed without prepared execution")
            cpu_result = self.cpu_executor.complete(cpu_prepared, cpu_host_result)
            if cpu_result.routed_pairs != cpu_pairs:
                raise AssertionError(
                    f"CPU executor handled {cpu_result.routed_pairs} routes, expected {cpu_pairs}"
                )
            output.index_add_(0, cpu_result.token_ids, cpu_result.output)
            self._record_cpu_result(metrics, cpu_result)

        if profile:
            metrics.host_dispatch_ms += (time.perf_counter_ns() - dispatch_started) / 1e6
        metrics.forward_calls += 1
        metrics.routed_pairs += total_pairs
        metrics.gpu_routed_pairs += total_pairs - cpu_pairs
        metrics.cpu_routed_pairs += cpu_pairs
        metrics.cache_hit_pairs += pair_hits
        metrics.cache_miss_pairs += total_pairs - pair_hits
        metrics.unique_miss_experts += len(plan.transfer_experts) + len(plan.cpu_experts)
        metrics.transfer_priority_experts += len(plan.transfer_experts)
        metrics.cpu_priority_experts += len(plan.cpu_experts)
        metrics.transfer_priority_sum += sum(
            expert_priorities[expert] for expert in plan.transfer_experts
        )
        metrics.cpu_priority_sum += sum(expert_priorities[expert] for expert in plan.cpu_experts)
        if profile:
            metrics.host_execute_ms += (time.perf_counter_ns() - execute_started) / 1e6
        return output

    @staticmethod
    def _record_cpu_result(metrics: LayerMetrics, result: CPUExecutionResult) -> None:
        metrics.cpu_active_experts += result.active_experts
        metrics.cpu_compact_tokens += result.compact_tokens
        metrics.cpu_single_token_experts += result.single_token_experts
        metrics.cpu_small_experts += result.small_experts
        metrics.cpu_max_tokens_per_expert = max(
            metrics.cpu_max_tokens_per_expert,
            result.max_tokens_per_expert,
        )
        metrics.cpu_workspace_allocations += result.workspace_allocations
        metrics.cpu_input_bytes += result.input_bytes
        metrics.cpu_output_bytes += result.output_bytes
        metrics.host_cpu_input_ms += result.input_copy_ms
        metrics.host_cpu_route_ms += result.route_ms
        metrics.host_cpu_compute_ms += result.compute_ms
        metrics.host_cpu_gate_up_ms += result.gate_up_ms
        metrics.host_cpu_activation_ms += result.activation_ms
        metrics.host_cpu_down_ms += result.down_ms
        metrics.host_cpu_scatter_ms += result.scatter_ms
        metrics.host_cpu_output_ms += result.output_copy_ms

    def _execute_static_gpu(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        gate_weights: torch.Tensor,
        execute_started: int,
    ) -> torch.Tensor:
        """Execute misses from scratch GPU storage without changing static residents."""

        profile = self.config.profiling.enabled
        device = hidden_states.device
        metrics = self._metrics[layer_id]
        self._prepare_static_layer(layer_id, device)
        placement = self._placement(layer_id)
        cache = self._cache(layer_id, device)

        route_started = time.perf_counter_ns()
        flat_experts = expert_ids.reshape(-1)
        unique_tensor, count_tensor = torch.unique(
            flat_experts,
            sorted=True,
            return_counts=True,
        )
        route_rows = torch.stack((unique_tensor, count_tensor), dim=1).cpu().tolist()
        unique_experts = [int(expert) for expert, _ in route_rows]
        route_counts = {int(expert): int(count) for expert, count in route_rows}
        resident = set(placement.gpu_experts)
        hit_experts = tuple(expert for expert in unique_experts if expert in resident)
        miss_experts = tuple(expert for expert in unique_experts if expert not in resident)
        if profile:
            metrics.host_route_plan_ms += (time.perf_counter_ns() - route_started) / 1e6

        self._record_cache_quality(
            metrics,
            active_experts=len(unique_experts),
            hit_experts=len(hit_experts),
            capacity=cache.capacity,
            occupied_slots=len(cache.resident_experts),
        )

        dispatch_started = time.perf_counter_ns()
        total_pairs = int(flat_experts.numel())
        pair_hits = sum(route_counts[expert] for expert in hit_experts)
        gpu_route_output = torch.zeros(
            (total_pairs, hidden_states.shape[-1]),
            device=device,
            dtype=hidden_states.dtype,
        )

        transient_buffer = None
        transient_ticket = None
        if miss_experts:
            transient_buffer = self._transient_buffer(device)
            transient_ticket = self._begin_transient_transfer(
                layer_id=layer_id,
                buffer=transient_buffer,
                metrics=metrics,
                experts=list(miss_experts),
                store=cache.store,
                profile=profile,
            )

        hit_execution = self._execute_chunk(
            cache=cache,
            metrics=metrics,
            hidden_states=hidden_states,
            expert_ids=expert_ids,
            gate_weights=gate_weights,
            chunk=list(hit_experts),
            route_counts=route_counts,
            output=gpu_route_output,
            profile=profile,
        )
        if hit_execution is not None:
            self._add_timing(
                layer_id,
                "cuda_moe_compute_ms",
                hit_execution.timing_interval,
            )

        if transient_ticket is not None:
            if transient_buffer is None:
                raise AssertionError("static GPU miss transfer lost its buffer")
            wait_interval = transient_buffer.wait_for_transfer(
                transient_ticket,
                profile=profile,
            )
            self._add_timing(layer_id, "cuda_transfer_wait_ms", wait_interval)
            miss_execution = self._execute_chunk(
                cache=transient_buffer,
                metrics=metrics,
                hidden_states=hidden_states,
                expert_ids=expert_ids,
                gate_weights=gate_weights,
                chunk=list(miss_experts),
                route_counts=route_counts,
                output=gpu_route_output,
                profile=profile,
            )
            if miss_execution is None:
                raise AssertionError("static GPU miss chunk had no routed tokens")
            transient_buffer.mark_used(miss_execution.done_event)
            self._add_timing(
                layer_id,
                "cuda_moe_compute_ms",
                miss_execution.timing_interval,
            )

        output = gpu_route_output.view(
            expert_ids.shape[0],
            expert_ids.shape[1],
            hidden_states.shape[-1],
        ).sum(dim=1)
        if cache.resident_experts != resident:
            raise AssertionError("static GPU execution changed the resident expert set")
        if profile:
            metrics.host_dispatch_ms += (time.perf_counter_ns() - dispatch_started) / 1e6
        metrics.forward_calls += 1
        metrics.routed_pairs += total_pairs
        metrics.gpu_routed_pairs += total_pairs
        metrics.cache_hit_pairs += pair_hits
        metrics.cache_miss_pairs += total_pairs - pair_hits
        metrics.unique_miss_experts += len(miss_experts)
        if profile:
            metrics.host_execute_ms += (time.perf_counter_ns() - execute_started) / 1e6
        return output

    def _execute_static(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        gate_weights: torch.Tensor,
        execute_started: int,
    ) -> torch.Tensor:
        profile = self.config.profiling.enabled
        device = hidden_states.device
        metrics = self._metrics[layer_id]
        placement = self._placement(layer_id)
        if placement.gpu_experts:
            self._prepare_static_layer(layer_id, device)

        route_started = time.perf_counter_ns()
        flat_experts = expert_ids.reshape(-1)
        unique_tensor, count_tensor = torch.unique(
            flat_experts,
            sorted=True,
            return_counts=True,
        )
        route_rows = torch.stack((unique_tensor, count_tensor), dim=1).cpu().tolist()
        unique_experts = [int(expert) for expert, _ in route_rows]
        route_counts = {int(expert): int(count) for expert, count in route_rows}
        gpu_owned = set(placement.gpu_experts)
        gpu_experts = tuple(expert for expert in unique_experts if expert in gpu_owned)
        cpu_experts = tuple(expert for expert in unique_experts if expert not in gpu_owned)
        if profile:
            metrics.host_route_plan_ms += (time.perf_counter_ns() - route_started) / 1e6

        dispatch_started = time.perf_counter_ns()
        output = torch.zeros_like(hidden_states)
        total_pairs = int(flat_experts.numel())
        gpu_route_output = torch.zeros(
            (total_pairs, hidden_states.shape[-1]),
            device=device,
            dtype=hidden_states.dtype,
        )
        cpu_pairs = sum(route_counts[expert] for expert in cpu_experts)
        gpu_pairs = sum(route_counts[expert] for expert in gpu_experts)
        cache_capacity = 0
        occupied_slots = 0
        if placement.gpu_experts:
            cache = self._cache(layer_id, device)
            cache_capacity = cache.capacity
            occupied_slots = len(cache.resident_experts)
        self._record_cache_quality(
            metrics,
            active_experts=len(unique_experts),
            hit_experts=len(gpu_experts),
            capacity=cache_capacity,
            occupied_slots=occupied_slots,
        )

        if cpu_experts:
            if self.cpu_executor is None:
                raise AssertionError("static placement did not initialize the CPU executor")
            cpu_result = self.cpu_executor.execute(
                store=self.store.layer(layer_id),
                hidden_states=hidden_states,
                expert_ids=expert_ids,
                gate_weights=gate_weights,
                active_experts=cpu_experts,
                route_counts=route_counts,
                profile=profile,
            )
            if cpu_result.routed_pairs != cpu_pairs:
                raise AssertionError(
                    f"CPU executor handled {cpu_result.routed_pairs} routes, expected {cpu_pairs}"
                )
            output.index_add_(0, cpu_result.token_ids, cpu_result.output)
            self._record_cpu_result(metrics, cpu_result)

        if gpu_experts:
            cache = self._cache(layer_id, device)
            cache.touch(gpu_experts)
            gpu_execution = self._execute_chunk(
                cache=cache,
                metrics=metrics,
                hidden_states=hidden_states,
                expert_ids=expert_ids,
                gate_weights=gate_weights,
                chunk=list(gpu_experts),
                route_counts=route_counts,
                output=gpu_route_output,
                profile=profile,
            )
            if gpu_execution is None:
                raise AssertionError("static GPU expert chunk had no routed tokens")
            self._add_timing(
                layer_id,
                "cuda_moe_compute_ms",
                gpu_execution.timing_interval,
            )

        output.add_(
            gpu_route_output.view(
                expert_ids.shape[0],
                expert_ids.shape[1],
                hidden_states.shape[-1],
            ).sum(dim=1)
        )

        if cpu_pairs + gpu_pairs != total_pairs:
            raise AssertionError("static placement did not account for every routed pair")
        if profile:
            metrics.host_dispatch_ms += (time.perf_counter_ns() - dispatch_started) / 1e6
        metrics.forward_calls += 1
        metrics.routed_pairs += total_pairs
        metrics.gpu_routed_pairs += gpu_pairs
        metrics.cpu_routed_pairs += cpu_pairs
        metrics.cache_hit_pairs += gpu_pairs
        metrics.cache_miss_pairs += cpu_pairs
        metrics.unique_miss_experts += len(cpu_experts)
        if profile:
            metrics.host_execute_ms += (time.perf_counter_ns() - execute_started) / 1e6
        return output

    @staticmethod
    def _pipeline_chunk_size(*, remaining: int, replaceable: int, capacity: int) -> int:
        if remaining <= 0 or replaceable <= 0:
            return 0
        if remaining <= replaceable:
            return remaining
        return min(replaceable, max(1, capacity // 2))

    def _priority_update_interval(self, num_experts: int) -> int:
        configured = self.config.cache.update_interval
        if configured:
            return configured
        cache_ratio = self.config.cache.capacity_per_layer / num_experts
        if cache_ratio <= 0.10:
            return 8
        if cache_ratio <= 0.50:
            return 16
        return 24

    def _priority_enabled_for_layer(self, layer_id: int) -> bool:
        """Select layers with the largest cache miss opportunity using prior data."""

        if not self.config.cache.selective_priority:
            self._layer_priority_enabled[layer_id] = True
            return True
        if self._layer_priority_observations.get(layer_id, 0) == 0:
            self._layer_priority_enabled[layer_id] = True
            return True

        observed = [
            (opportunity, observed_layer)
            for observed_layer, opportunity in self._layer_priority_opportunity_ema.items()
            if self._layer_priority_observations.get(observed_layer, 0) > 0
        ]
        eligible_count = max(
            1,
            math.ceil(len(observed) * self.config.cache.priority_layer_fraction),
        )
        observed.sort(key=lambda row: (-row[0], row[1]))
        top_layers = {layer for _, layer in observed[:eligible_count]}
        boundary = observed[min(eligible_count, len(observed)) - 1][0]
        opportunity = self._layer_priority_opportunity_ema[layer_id]
        enabled = opportunity >= self.config.cache.min_priority_opportunity and (
            layer_id in top_layers
            or (
                self._layer_priority_enabled.get(layer_id, False)
                and opportunity + self.config.cache.priority_hysteresis >= boundary
            )
        )
        self._layer_priority_enabled[layer_id] = enabled
        return enabled

    def _observe_layer_priority_opportunity(
        self,
        *,
        layer_id: int,
        active_experts: int,
        hit_experts: int,
        capacity: int,
    ) -> float:
        """Update the EMA of cache slots whose misses a better policy could avoid."""

        recoverable_misses = max(0, min(active_experts, capacity) - hit_experts)
        sample = recoverable_misses / capacity
        observations = self._layer_priority_observations.get(layer_id, 0)
        if observations:
            decay = self.config.cache.utilization_ema_decay
            previous = self._layer_priority_opportunity_ema[layer_id]
            ema = decay * previous + (1.0 - decay) * sample
        else:
            ema = sample
        self._layer_priority_opportunity_ema[layer_id] = ema
        self._layer_priority_observations[layer_id] = observations + 1
        return ema

    def _begin_transfer(
        self,
        *,
        layer_id: int,
        cache: LayerDeviceCache,
        metrics: LayerMetrics,
        experts: list[int],
        protected: Iterable[int],
        slots_reusable_after: torch.cuda.Event | None,
        eviction_buffer: DeviceEvictionBuffer | None,
        capture_experts: Iterable[int],
        profile: bool,
    ) -> TransferTicket:
        ticket = cache.begin_transfer(
            experts,
            protected_experts=protected,
            slots_reusable_after=slots_reusable_after,
            eviction_buffer=eviction_buffer,
            capture_experts=capture_experts,
            profile=profile,
        )
        metrics.h2d_expert_loads += ticket.load_count
        metrics.h2d_copy_submissions += ticket.copy_submissions
        metrics.h2d_bytes += ticket.loaded_bytes
        self._add_timing(layer_id, "cuda_h2d_ms", ticket.dma_interval)
        return ticket

    @staticmethod
    def _record_cache_quality(
        metrics: LayerMetrics,
        *,
        active_experts: int,
        hit_experts: int,
        capacity: int,
        occupied_slots: int,
    ) -> None:
        metrics.active_experts += active_experts
        metrics.cache_hit_experts += hit_experts
        metrics.cache_slot_observations += capacity
        metrics.cache_occupied_slot_observations += occupied_slots
        metrics.priority_opportunity_slots += max(
            0,
            min(active_experts, capacity) - hit_experts,
        )

    def _await_retention(
        self,
        layer_id: int,
        cache: LayerDeviceCache,
        metrics: LayerMetrics,
        *,
        profile: bool,
    ) -> None:
        ticket = self._retention_tickets.pop(layer_id, None)
        if ticket is None:
            return
        wait_interval = cache.wait_for_transfer(ticket, profile=profile)
        self._add_timing(layer_id, "cuda_retention_wait_ms", wait_interval)

    def _schedule_policy_retention(
        self,
        *,
        layer_id: int,
        cache: LayerDeviceCache,
        metrics: LayerMetrics,
        retained: tuple[int, ...],
        eviction_buffer: DeviceEvictionBuffer | None,
        slots_reusable_after: torch.cuda.Event | None,
        profile: bool,
    ) -> None:
        if eviction_buffer is not None:
            capture_count = len(eviction_buffer.expert_to_slot)
            metrics.retention_capture_device_copies += capture_count
            metrics.retention_capture_d2d_bytes += capture_count * cache.store.expert_bytes
        resident_now = cache.resident_experts
        missing = [expert for expert in retained if expert not in resident_now]
        if not missing:
            cache.touch(retained)
            return
        if eviction_buffer is None:
            raise AssertionError("policy retention needs an eviction buffer")
        protected = [expert for expert in retained if expert in resident_now]
        ticket = cache.begin_restore(
            missing,
            eviction_buffer=eviction_buffer,
            protected_experts=protected,
            slots_reusable_after=slots_reusable_after,
            profile=profile,
        )
        metrics.retention_device_copies += ticket.copy_submissions
        metrics.retention_d2d_bytes += ticket.loaded_bytes
        self._add_timing(layer_id, "cuda_retention_d2d_ms", ticket.dma_interval)
        self._retention_tickets[layer_id] = ticket

    def _begin_transient_transfer(
        self,
        *,
        layer_id: int,
        buffer: DeviceTransientExpertBuffer,
        metrics: LayerMetrics,
        experts: list[int],
        store,
        profile: bool,
    ) -> TransferTicket:
        ticket = buffer.begin_transfer(experts, store=store, profile=profile)
        metrics.h2d_expert_loads += ticket.load_count
        metrics.transient_expert_loads += ticket.load_count
        metrics.h2d_copy_submissions += ticket.copy_submissions
        metrics.h2d_bytes += ticket.loaded_bytes
        self._add_timing(layer_id, "cuda_h2d_ms", ticket.dma_interval)
        return ticket

    def _execute_chunk(
        self,
        *,
        cache: LayerDeviceCache | DeviceTransientExpertBuffer,
        fallback_cache: LayerDeviceCache | DeviceTransientExpertBuffer | None = None,
        metrics: LayerMetrics,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        gate_weights: torch.Tensor,
        chunk: list[int],
        route_counts: dict[int, int],
        output: torch.Tensor,
        profile: bool,
    ) -> _ChunkExecution | None:
        if not chunk:
            return None
        device = hidden_states.device
        expert_batch_size = self.config.transfer.gpu_expert_batch_size
        if len(chunk) > expert_batch_size:
            compute_start = None
            if profile:
                compute_start = torch.cuda.Event(enable_timing=True)
                compute_start.record(torch.cuda.current_stream(device))
            for offset in range(0, len(chunk), expert_batch_size):
                self._execute_chunk(
                    cache=cache,
                    fallback_cache=fallback_cache,
                    metrics=metrics,
                    hidden_states=hidden_states,
                    expert_ids=expert_ids,
                    gate_weights=gate_weights,
                    chunk=chunk[offset : offset + expert_batch_size],
                    route_counts=route_counts,
                    output=output,
                    profile=False,
                )
            compute_done = torch.cuda.Event(enable_timing=profile)
            compute_done.record(torch.cuda.current_stream(device))
            interval = (compute_start, compute_done) if compute_start is not None else None
            return _ChunkExecution(compute_done, interval)

        n_tokens, top_k = expert_ids.shape
        metrics.gpu_chunk_executions += 1
        metrics.gpu_chunk_experts += len(chunk)
        metrics.gpu_chunk_padded_experts += expert_batch_size - len(chunk)
        flat_experts = expert_ids.reshape(-1)
        flat_weights = gate_weights.reshape(-1).to(dtype=hidden_states.dtype)
        token_ids = torch.arange(n_tokens, device=device).repeat_interleave(top_k)
        chunk_tensor = torch.tensor(chunk, device=device, dtype=expert_ids.dtype)
        selected = torch.isin(flat_experts, chunk_tensor)
        selected_routes = torch.nonzero(selected, as_tuple=False).flatten()
        selected_experts = flat_experts[selected]
        selected_tokens = token_ids[selected]
        selected_weights = flat_weights[selected]

        lookup = torch.full(
            (cache.store.shape.num_experts,),
            -1,
            device=device,
            dtype=torch.long,
        )
        lookup[chunk_tensor.long()] = torch.arange(len(chunk), device=device)
        inverse = lookup[selected_experts.long()]
        order = torch.argsort(inverse)
        inverse_sorted = inverse[order]
        route_sorted = selected_routes[order]
        token_sorted = selected_tokens[order]
        weight_sorted = selected_weights[order]
        x_sorted = hidden_states[token_sorted]

        counts = torch.bincount(inverse_sorted, minlength=len(chunk))
        # Keep GEMM M dimensions independent of cache chunking so cache policy
        # cannot perturb BF16 results through a different padded matrix shape.
        max_tokens = max(route_counts.values())
        offsets = torch.cumsum(counts, dim=0) - counts
        positions = torch.arange(inverse_sorted.numel(), device=device) - offsets[inverse_sorted]

        compute_start = None
        if profile:
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_start.record(torch.cuda.current_stream(device))

        hidden_size = cache.store.shape.hidden_size
        intermediate_size = cache.store.shape.intermediate_size
        # Keep the BF16 bmm batch shape independent of cache hit/miss grouping.
        compute_experts = expert_batch_size
        batch_x = torch.zeros(
            (compute_experts, max_tokens, hidden_size),
            device=device,
            dtype=hidden_states.dtype,
        )
        batch_weight = torch.zeros(
            (compute_experts, max_tokens, 1), device=device, dtype=hidden_states.dtype
        )
        batch_x[inverse_sorted, positions] = x_sorted
        batch_weight[inverse_sorted, positions] = weight_sorted.unsqueeze(-1)

        if fallback_cache is None:
            slot_ids = [cache.slot(expert) for expert in chunk]
            slot_ids.extend([slot_ids[0]] * (compute_experts - len(slot_ids)))
            slots = torch.tensor(slot_ids, device=device)
            w13 = cache.w13.index_select(0, slots)
            w2 = cache.w2.index_select(0, slots)
        else:
            weight_sources = [
                cache if expert in cache.expert_to_slot else fallback_cache for expert in chunk
            ]
            if any(
                expert not in source.expert_to_slot
                for expert, source in zip(chunk, weight_sources, strict=True)
            ):
                raise KeyError("a mixed expert chunk is missing GPU weights")
            w13_rows = [
                source.w13[source.slot(expert)]
                for expert, source in zip(chunk, weight_sources, strict=True)
            ]
            w2_rows = [
                source.w2[source.slot(expert)]
                for expert, source in zip(chunk, weight_sources, strict=True)
            ]
            w13_rows.extend([w13_rows[0]] * (compute_experts - len(w13_rows)))
            w2_rows.extend([w2_rows[0]] * (compute_experts - len(w2_rows)))
            w13 = torch.stack(w13_rows)
            w2 = torch.stack(w2_rows)
        gate = torch.bmm(batch_x, w13[:, :intermediate_size].transpose(1, 2))
        up = torch.bmm(batch_x, w13[:, intermediate_size:].transpose(1, 2))
        activated = F.silu(gate) * up
        result = torch.bmm(activated, w2.transpose(1, 2)) * batch_weight
        result_sorted = result[inverse_sorted, positions].to(dtype=output.dtype)
        output[route_sorted] = result_sorted

        compute_done = torch.cuda.Event(enable_timing=profile)
        compute_done.record(torch.cuda.current_stream(device))
        interval = (compute_start, compute_done) if compute_start is not None else None
        return _ChunkExecution(compute_done, interval)

    def reset_metrics(self) -> None:
        if self._pending_timings or self._retention_tickets:
            torch.cuda.synchronize()
            self._pending_timings.clear()
            self._retention_tickets.clear()
        for layer_id in self._metrics:
            self._metrics[layer_id] = LayerMetrics()

    def close(self) -> None:
        if self._cache_trace_path:
            trace_path = os.path.abspath(self._cache_trace_path)
            trace_dir = os.path.dirname(trace_path)
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
            with open(trace_path, "w", encoding="utf-8") as handle:
                json.dump(self._cache_trace, handle)
            self._cache_trace_path = None
        if self._cpu_dispatch_pool is not None:
            self._cpu_dispatch_pool.shutdown(wait=True)
            self._cpu_dispatch_pool = None
        if self.cpu_executor is not None:
            self.cpu_executor.close()

    def metrics(self) -> dict[int, dict[str, int | float]]:
        self._drain_timings()
        result: dict[int, dict[str, int | float]] = {}
        capacity_metrics = self.layer_capacity_controller.layer_metrics()
        for layer_id, values in sorted(self._metrics.items()):
            layer = values.as_dict()
            layer.update(capacity_metrics.get(layer_id, {}))
            layer["priority_opportunity_ema"] = self._layer_priority_opportunity_ema.get(
                layer_id, 0.0
            )
            layer["priority_enabled"] = int(self._layer_priority_enabled.get(layer_id, False))
            result[layer_id] = layer
        return result

    def capacity_plan(self) -> dict[str, object]:
        """Return the M3.0 dry-run allocation state for reports and tests."""

        return self.layer_capacity_controller.summary()

    def summary(self) -> dict[str, int | float | bool | str]:
        self._drain_timings()
        layers = list(self._metrics.values())
        layer_ids = self.store.layer_ids
        effective_priority_interval = (
            self._priority_update_interval(
                self.store.layer(layer_ids[0]).shape.num_experts,
            )
            if layer_ids
            else self.config.cache.update_interval
        )
        pairs = sum(layer.routed_pairs for layer in layers)
        hits = sum(layer.cache_hit_pairs for layer in layers)
        active_experts = sum(layer.active_experts for layer in layers)
        hit_experts = sum(layer.cache_hit_experts for layer in layers)
        slot_observations = sum(layer.cache_slot_observations for layer in layers)
        forward_calls = sum(layer.forward_calls for layer in layers)
        h2d_bytes = sum(layer.h2d_bytes for layer in layers)
        cuda_h2d_ms = sum(layer.cuda_h2d_ms for layer in layers)
        capacity_summary = self.layer_capacity_controller.summary()
        transfer_priority_experts = sum(layer.transfer_priority_experts for layer in layers)
        cpu_priority_experts = sum(layer.cpu_priority_experts for layer in layers)
        return {
            "cache_policy": self.config.cache.policy,
            "execution_mode": self.config.placement.mode,
            "cpu_miss_ratio": self.config.placement.cpu_miss_ratio,
            "cpu_min_dispatch_experts": (
                self.config.placement.cpu_min_dispatch_experts
            ),
            "transfer_high_priority": self.config.placement.transfer_high_priority,
            "cpu_execution_strategy": self.config.cpu.strategy,
            "cpu_compact_io": self.config.cpu.compact_io,
            "profiling_enabled": self.config.profiling.enabled,
            "layer_capacity_dry_run": bool(capacity_summary["dry_run"]),
            "layer_capacity_apply": bool(capacity_summary["applied"]),
            "layer_capacity_completed_forwards": int(capacity_summary["completed_forwards"]),
            "layer_capacity_updates": int(capacity_summary["updates"]),
            "layer_capacity_applied_updates": self._applied_layer_capacity_updates,
            "layer_capacity_released_slots": self._layer_capacity_released_slots,
            "layer_capacity_skipped_forwards": self._layer_capacity_skipped_forwards,
            "layer_capacity_total_slots": int(capacity_summary["total_slots"]),
            "layer_capacity_last_moved_slots": int(capacity_summary["last_moved_slots"]),
            "layer_capacity_converged": bool(capacity_summary["converged"]),
            "layer_capacity_stable_updates": int(capacity_summary["stable_updates"]),
            "layer_capacity_current_update_interval": int(
                capacity_summary["current_update_interval"]
            ),
            "layer_capacity_next_update_forward": int(capacity_summary["next_update_forward"]),
            "host_layer_capacity_observe_ms": self._layer_capacity_observe_ns / 1e6,
            "host_layer_capacity_update_ms": self._layer_capacity_update_ns / 1e6,
            "host_layer_capacity_apply_ms": self._layer_capacity_apply_ns / 1e6,
            "layer_capacity_recommended_min": int(capacity_summary["recommended_min"]),
            "layer_capacity_recommended_max": int(capacity_summary["recommended_max"]),
            "layers": len(layers),
            "forward_calls": forward_calls,
            "routed_pairs": pairs,
            "cache_hit_pairs": hits,
            "cache_miss_pairs": sum(layer.cache_miss_pairs for layer in layers),
            "cache_hit_rate": hits / pairs if pairs else 0.0,
            "active_experts": active_experts,
            "cache_hit_experts": hit_experts,
            "expert_cache_hit_rate": (hit_experts / active_experts if active_experts else 0.0),
            "cache_slot_observations": slot_observations,
            "cache_utilization": (hit_experts / slot_observations if slot_observations else 0.0),
            "cache_occupancy": (
                sum(layer.cache_occupied_slot_observations for layer in layers) / slot_observations
                if slot_observations
                else 0.0
            ),
            "priority_opportunity": (
                sum(layer.priority_opportunity_slots for layer in layers) / slot_observations
                if slot_observations
                else 0.0
            ),
            "gpu_routed_pairs": sum(layer.gpu_routed_pairs for layer in layers),
            "gpu_chunk_executions": sum(layer.gpu_chunk_executions for layer in layers),
            "gpu_chunk_experts": sum(layer.gpu_chunk_experts for layer in layers),
            "gpu_chunk_padded_experts": sum(layer.gpu_chunk_padded_experts for layer in layers),
            "cpu_routed_pairs": sum(layer.cpu_routed_pairs for layer in layers),
            "cpu_active_experts": sum(layer.cpu_active_experts for layer in layers),
            "cpu_compact_tokens": sum(layer.cpu_compact_tokens for layer in layers),
            "cpu_single_token_experts": sum(layer.cpu_single_token_experts for layer in layers),
            "cpu_small_experts": sum(layer.cpu_small_experts for layer in layers),
            "cpu_max_tokens_per_expert": max(
                (layer.cpu_max_tokens_per_expert for layer in layers),
                default=0,
            ),
            "cpu_workspace_allocations": sum(layer.cpu_workspace_allocations for layer in layers),
            "cpu_dispatch_submissions": sum(layer.cpu_dispatch_submissions for layer in layers),
            "unique_miss_experts": sum(layer.unique_miss_experts for layer in layers),
            "transfer_priority_experts": transfer_priority_experts,
            "cpu_priority_experts": cpu_priority_experts,
            "transfer_mean_priority": (
                sum(layer.transfer_priority_sum for layer in layers) / transfer_priority_experts
                if transfer_priority_experts
                else 0.0
            ),
            "cpu_mean_priority": (
                sum(layer.cpu_priority_sum for layer in layers) / cpu_priority_experts
                if cpu_priority_experts
                else 0.0
            ),
            "h2d_expert_loads": sum(layer.h2d_expert_loads for layer in layers),
            "transient_expert_loads": sum(layer.transient_expert_loads for layer in layers),
            "h2d_copy_submissions": sum(layer.h2d_copy_submissions for layer in layers),
            "h2d_bytes": h2d_bytes,
            "retention_device_copies": sum(layer.retention_device_copies for layer in layers),
            "retention_d2d_bytes": sum(layer.retention_d2d_bytes for layer in layers),
            "retention_capture_device_copies": sum(
                layer.retention_capture_device_copies for layer in layers
            ),
            "retention_capture_d2d_bytes": sum(
                layer.retention_capture_d2d_bytes for layer in layers
            ),
            "priority_updates": sum(layer.priority_updates for layer in layers),
            "priority_update_interval": effective_priority_interval,
            "priority_update_rate": (
                sum(layer.priority_updates for layer in layers) / forward_calls
                if forward_calls
                else 0.0
            ),
            "priority_scheduled_calls": sum(layer.priority_scheduled_calls for layer in layers),
            "priority_eligible_calls": sum(layer.priority_eligible_calls for layer in layers),
            "priority_bypassed_calls": sum(layer.priority_bypassed_calls for layer in layers),
            "route_accounting_calls": sum(layer.route_accounting_calls for layer in layers),
            "route_accounted_pairs": sum(layer.route_accounted_pairs for layer in layers),
            "retention_updates": sum(layer.retention_updates for layer in layers),
            "route_tracked_experts": sum(
                int(np.count_nonzero(frequency)) for frequency in self._route_frequency.values()
            ),
            "route_max_frequency": max(
                (int(frequency.max()) for frequency in self._route_frequency.values()),
                default=0,
            ),
            "priority_enabled_layers": sum(self._layer_priority_enabled.values()),
            "priority_observed_layers": len(self._layer_priority_observations),
            "selective_priority": self.config.cache.selective_priority,
            "priority_layer_fraction": self.config.cache.priority_layer_fraction,
            "cpu_input_bytes": sum(layer.cpu_input_bytes for layer in layers),
            "cpu_output_bytes": sum(layer.cpu_output_bytes for layer in layers),
            "cuda_h2d_ms": cuda_h2d_ms,
            "cuda_transfer_wait_ms": sum(layer.cuda_transfer_wait_ms for layer in layers),
            "cuda_retention_wait_ms": sum(layer.cuda_retention_wait_ms for layer in layers),
            "cuda_retention_d2d_ms": sum(layer.cuda_retention_d2d_ms for layer in layers),
            "cuda_priority_ms": sum(layer.cuda_priority_ms for layer in layers),
            "cuda_moe_compute_ms": sum(layer.cuda_moe_compute_ms for layer in layers),
            "host_execute_ms": sum(layer.host_execute_ms for layer in layers),
            "host_route_plan_ms": sum(layer.host_route_plan_ms for layer in layers),
            "host_cache_stats_ms": sum(layer.host_cache_stats_ms for layer in layers),
            "host_priority_ms": sum(layer.host_priority_ms for layer in layers),
            "host_dispatch_ms": sum(layer.host_dispatch_ms for layer in layers),
            "host_cpu_input_ms": sum(layer.host_cpu_input_ms for layer in layers),
            "host_cpu_route_ms": sum(layer.host_cpu_route_ms for layer in layers),
            "host_cpu_compute_ms": sum(layer.host_cpu_compute_ms for layer in layers),
            "host_cpu_gate_up_ms": sum(layer.host_cpu_gate_up_ms for layer in layers),
            "host_cpu_activation_ms": sum(layer.host_cpu_activation_ms for layer in layers),
            "host_cpu_down_ms": sum(layer.host_cpu_down_ms for layer in layers),
            "host_cpu_scatter_ms": sum(layer.host_cpu_scatter_ms for layer in layers),
            "host_cpu_output_ms": sum(layer.host_cpu_output_ms for layer in layers),
            "host_cpu_wait_ms": sum(layer.host_cpu_wait_ms for layer in layers),
            "h2d_effective_gbps": (h2d_bytes / cuda_h2d_ms / 1_000_000 if cuda_h2d_ms else 0.0),
            "host_expert_bytes": self.store.allocated_bytes,
            "gpu_cache_bytes": sum(pool.allocated_bytes for pool in self._device_pools.values()),
            "gpu_cache_pool_count": len(self._device_pools),
            "gpu_cache_pool_free_slots": sum(
                pool.free_slots for pool in self._device_pools.values()
            ),
            "gpu_eviction_buffer_bytes": sum(
                buffer.allocated_bytes for buffer in self._eviction_buffers.values()
            ),
            "gpu_transient_buffer_bytes": sum(
                buffer.allocated_bytes for buffer in self._transient_buffers.values()
            ),
        }
