from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from oled_moe.memory import LayerExpertStore


@dataclass(frozen=True)
class CPUExecutionResult:
    """A compact CPU contribution and its measured execution characteristics."""

    output: torch.Tensor
    token_ids: torch.Tensor
    active_experts: int
    routed_pairs: int
    compact_tokens: int
    single_token_experts: int
    small_experts: int
    max_tokens_per_expert: int
    workspace_allocations: int
    input_bytes: int
    output_bytes: int
    input_copy_ms: float
    route_ms: float
    compute_ms: float
    gate_up_ms: float
    activation_ms: float
    down_ms: float
    scatter_ms: float
    output_copy_ms: float


@dataclass
class _CPUWorkspace:
    hidden: torch.Tensor
    output: torch.Tensor
    route_experts: torch.Tensor
    route_weights: torch.Tensor
    token_capacity: int
    route_capacity: int
    output_reusable_event: torch.cuda.Event | None = None


@dataclass(frozen=True)
class _ExpertJob:
    expert_id: int
    source_tokens: torch.Tensor
    output_tokens: torch.Tensor
    weights: torch.Tensor
    hidden: torch.Tensor
    store: LayerExpertStore
    profile: bool


@dataclass(frozen=True)
class _ExpertResult:
    output_tokens: torch.Tensor
    output: torch.Tensor
    gate_up_ms: float
    activation_ms: float
    down_ms: float


@dataclass(frozen=True)
class CPUPreparedExecution:
    jobs: tuple[_ExpertJob, ...]
    output_cpu: torch.Tensor
    output_tokens: torch.Tensor
    output_device: torch.device
    output_dtype: torch.dtype
    active_experts: int
    routed_pairs: int
    compact_tokens: int
    single_token_experts: int
    small_experts: int
    max_tokens_per_expert: int
    workspace_allocations: int
    input_bytes: int
    output_bytes: int
    input_copy_ms: float
    route_ms: float
    profile: bool
    device_token_ids: torch.Tensor | None = None
    workspace: _CPUWorkspace | None = None
    workspace_ready: torch.cuda.Event | None = None
    input_ready: torch.cuda.Event | None = None
    input_timing: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    input_sources: tuple[torch.Tensor, ...] = ()


def _capacity(required: int) -> int:
    """Grow reusable buffers geometrically to avoid reallocating for nearby shapes."""

    return 1 if required <= 1 else 1 << (required - 1).bit_length()


class CPUExpertExecutor:
    """Synchronous CPU baseline with compact routes and reusable host workspaces."""

    def __init__(
        self,
        *,
        num_threads: int,
        strategy: str = "expert_parallel",
        pointwise_mode: str = "inplace",
    ) -> None:
        if strategy not in {"serial", "expert_parallel"}:
            raise ValueError(f"unsupported CPU execution strategy: {strategy}")
        if pointwise_mode not in {"baseline", "inplace"}:
            raise ValueError(f"unsupported CPU pointwise mode: {pointwise_mode}")
        self.num_threads = num_threads
        self.strategy = strategy
        self.pointwise_mode = pointwise_mode
        torch.set_num_threads(num_threads if strategy == "serial" else 1)
        self._pool = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads,
                thread_name_prefix="oled-cpu-expert",
            )
            if strategy == "expert_parallel"
            else None
        )
        self._workspaces: dict[tuple[int | None, int, torch.dtype, bool], _CPUWorkspace] = {}
        self._workspace_allocations = 0
        self._route_lookups: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}
        self._token_ids: dict[tuple[int, int], torch.Tensor] = {}
        self._input_streams: dict[int, torch.cuda.Stream] = {}
        self._result_streams: dict[int, torch.cuda.Stream] = {}

    @property
    def workspace_allocations(self) -> int:
        return self._workspace_allocations

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for workspace in self._workspaces.values():
            if workspace.output_reusable_event is not None:
                workspace.output_reusable_event.synchronize()
                workspace.output_reusable_event = None
        for stream in (*self._input_streams.values(), *self._result_streams.values()):
            stream.synchronize()

    def _compute_expert(self, job: _ExpertJob) -> _ExpertResult:
        intermediate_size = job.store.shape.intermediate_size
        selected_x = job.hidden.index_select(0, job.source_tokens)

        stage_started = time.perf_counter_ns()
        gate_up = F.linear(selected_x, job.store.w13[job.expert_id])
        gate_up_ms = (
            (time.perf_counter_ns() - stage_started) / 1e6 if job.profile else 0.0
        )

        stage_started = time.perf_counter_ns()
        gate = gate_up[:, :intermediate_size]
        up = gate_up[:, intermediate_size:]
        if self.pointwise_mode == "inplace":
            F.silu(gate, inplace=True)
            gate.mul_(up)
            activated = gate
        else:
            activated = F.silu(gate) * up
        activation_ms = (
            (time.perf_counter_ns() - stage_started) / 1e6 if job.profile else 0.0
        )

        stage_started = time.perf_counter_ns()
        output = F.linear(activated, job.store.w2[job.expert_id])
        down_ms = (
            (time.perf_counter_ns() - stage_started) / 1e6 if job.profile else 0.0
        )
        if self.pointwise_mode == "inplace":
            output.mul_(job.weights.unsqueeze(-1))
        else:
            output = output * job.weights.unsqueeze(-1)
        return _ExpertResult(
            output_tokens=job.output_tokens,
            output=output,
            gate_up_ms=gate_up_ms,
            activation_ms=activation_ms,
            down_ms=down_ms,
        )

    def _workspace(
        self,
        *,
        hidden_size: int,
        dtype: torch.dtype,
        pin_memory: bool,
        tokens: int,
        routes: int,
        scope: int | None = None,
    ) -> tuple[_CPUWorkspace, int]:
        key = (scope, hidden_size, dtype, pin_memory)
        existing = self._workspaces.get(key)
        if (
            existing is not None
            and existing.token_capacity >= tokens
            and existing.route_capacity >= routes
        ):
            return existing, 0
        if existing is not None and existing.output_reusable_event is not None:
            # A growing workspace replaces the last Python reference to its
            # pinned output buffer. Keep that source alive until its async H2D
            # copy completes; growth is rare and occurs during warmup.
            existing.output_reusable_event.synchronize()
            existing.output_reusable_event = None

        token_capacity = _capacity(
            max(tokens, existing.token_capacity if existing is not None else 0)
        )
        route_capacity = _capacity(
            max(routes, existing.route_capacity if existing is not None else 0)
        )
        workspace = _CPUWorkspace(
            hidden=torch.empty(
                (token_capacity, hidden_size),
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
            ),
            output=torch.empty(
                (token_capacity, hidden_size),
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
            ),
            route_experts=torch.empty(
                route_capacity,
                dtype=torch.long,
                device="cpu",
                pin_memory=pin_memory,
            ),
            route_weights=torch.empty(
                route_capacity,
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
            ),
            token_capacity=token_capacity,
            route_capacity=route_capacity,
        )
        self._workspaces[key] = workspace
        self._workspace_allocations += 1
        return workspace, 1

    @staticmethod
    def _device_index(device: torch.device) -> int:
        return device.index if device.index is not None else torch.cuda.current_device()

    def _input_stream(self, device: torch.device) -> torch.cuda.Stream:
        device_index = self._device_index(device)
        stream = self._input_streams.get(device_index)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._input_streams[device_index] = stream
        return stream

    def _result_stream(self, device: torch.device) -> torch.cuda.Stream:
        device_index = self._device_index(device)
        stream = self._result_streams.get(device_index)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._result_streams[device_index] = stream
        return stream

    def _route_lookup(
        self,
        *,
        num_experts: int,
        active_experts: tuple[int, ...],
    ) -> torch.Tensor:
        key = (num_experts, active_experts)
        existing = self._route_lookups.get(key)
        if existing is not None:
            return existing
        lookup = torch.full((num_experts,), -1, dtype=torch.long, device="cpu")
        expert_tensor = torch.tensor(active_experts, dtype=torch.long, device="cpu")
        lookup[expert_tensor] = torch.arange(len(active_experts))
        self._route_lookups[key] = lookup
        return lookup

    def _flat_token_ids(
        self,
        *,
        n_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        key = (n_tokens, top_k)
        existing = self._token_ids.get(key)
        if existing is not None:
            return existing
        token_ids = torch.arange(n_tokens).repeat_interleave(top_k)
        self._token_ids[key] = token_ids
        return token_ids

    @torch.no_grad()
    def prepare(
        self,
        *,
        store: LayerExpertStore,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        gate_weights: torch.Tensor,
        active_experts: tuple[int, ...],
        route_counts: dict[int, int],
        profile: bool,
    ) -> CPUPreparedExecution:
        if not active_experts:
            raise ValueError("CPU execution requires at least one active expert")
        active_experts = tuple(sorted(active_experts))
        counts = tuple(route_counts.get(expert, 0) for expert in active_experts)
        if any(count <= 0 for count in counts):
            raise ValueError("active CPU experts must all own at least one route")
        routed_pairs = sum(counts)

        input_started = time.perf_counter_ns()
        n_tokens, top_k = expert_ids.shape
        total_routes = int(expert_ids.numel())
        use_pinned_workspace = store.pin_memory and torch.cuda.is_available()
        workspace, workspace_allocations = self._workspace(
            hidden_size=store.shape.hidden_size,
            dtype=store.dtype,
            pin_memory=use_pinned_workspace,
            tokens=n_tokens,
            routes=total_routes,
        )
        hidden_cpu = workspace.hidden[:n_tokens]
        route_experts_cpu = workspace.route_experts[:total_routes]
        route_weights_cpu = workspace.route_weights[:total_routes]
        hidden_cpu.copy_(hidden_states.to(dtype=store.dtype))
        route_experts_cpu.copy_(expert_ids.reshape(-1))
        route_weights_cpu.copy_(gate_weights.reshape(-1).to(dtype=store.dtype))
        input_copy_ms = (
            (time.perf_counter_ns() - input_started) / 1e6 if profile else 0.0
        )

        route_started = time.perf_counter_ns()
        token_ids = self._flat_token_ids(n_tokens=n_tokens, top_k=top_k)
        lookup = self._route_lookup(
            num_experts=store.shape.num_experts,
            active_experts=active_experts,
        )
        route_groups = lookup[route_experts_cpu]
        selected = route_groups >= 0
        selected_groups = route_groups[selected]
        selected_tokens = token_ids[selected]
        selected_weights = route_weights_cpu[selected]
        order = torch.argsort(selected_groups, stable=True)
        token_sorted = selected_tokens[order]
        weight_sorted = selected_weights[order]
        unique_tokens, compact_route_tokens = torch.unique(
            token_sorted,
            sorted=True,
            return_inverse=True,
        )
        compact_tokens = int(unique_tokens.numel())
        if routed_pairs != int(token_sorted.numel()):
            raise AssertionError(
                f"compact route plan contains {token_sorted.numel()} routes, "
                f"expected {routed_pairs}"
            )
        route_ms = (time.perf_counter_ns() - route_started) / 1e6 if profile else 0.0

        output_cpu = workspace.output[:compact_tokens]
        output_cpu.zero_()
        offsets: list[int] = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)

        jobs = []
        for group_id, expert_id in enumerate(active_experts):
            start = offsets[group_id]
            end = offsets[group_id + 1]
            jobs.append(
                _ExpertJob(
                    expert_id=expert_id,
                    source_tokens=token_sorted[start:end],
                    output_tokens=compact_route_tokens[start:end],
                    weights=weight_sorted[start:end],
                    hidden=hidden_cpu,
                    store=store,
                    profile=profile,
                )
            )

        input_bytes = (
            hidden_cpu.numel() * hidden_cpu.element_size()
            + route_experts_cpu.numel() * route_experts_cpu.element_size()
            + route_weights_cpu.numel() * route_weights_cpu.element_size()
        )
        output_bytes = output_cpu.numel() * output_cpu.element_size()
        return CPUPreparedExecution(
            jobs=tuple(jobs),
            output_cpu=output_cpu,
            output_tokens=unique_tokens,
            output_device=hidden_states.device,
            output_dtype=hidden_states.dtype,
            active_experts=len(active_experts),
            routed_pairs=routed_pairs,
            compact_tokens=compact_tokens,
            single_token_experts=sum(count == 1 for count in counts),
            small_experts=sum(count <= 4 for count in counts),
            max_tokens_per_expert=max(counts),
            workspace_allocations=workspace_allocations,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            input_copy_ms=input_copy_ms,
            route_ms=route_ms,
            profile=profile,
        )

    @torch.no_grad()
    def prepare_compact(
        self,
        *,
        store: LayerExpertStore,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        route_packet_host: np.ndarray,
        active_experts: tuple[int, ...],
        route_counts: dict[int, int],
        profile: bool,
        workspace_scope: int | None = None,
    ) -> CPUPreparedExecution:
        """Pack only CPU-routed tokens using the route packet already on host."""

        if not active_experts:
            raise ValueError("CPU execution requires at least one active expert")
        if route_packet_host.ndim != 2 or route_packet_host.shape[1] < 3:
            raise ValueError(
                "compact CPU execution requires route IDs, priorities, and gates"
            )
        active_experts = tuple(sorted(active_experts))
        counts = tuple(route_counts.get(expert, 0) for expert in active_experts)
        if any(count <= 0 for count in counts):
            raise ValueError("active CPU experts must all own at least one route")
        routed_pairs = sum(counts)

        route_started = time.perf_counter_ns()
        n_tokens, top_k = expert_ids.shape
        total_routes = n_tokens * top_k
        route_experts = route_packet_host[:, 0].astype(np.int64, copy=False)
        if route_experts.size != total_routes:
            raise ValueError("host route packet does not match the routed tensor")
        lookup = np.full(store.shape.num_experts, -1, dtype=np.int64)
        lookup[np.asarray(active_experts, dtype=np.int64)] = np.arange(
            len(active_experts),
            dtype=np.int64,
        )
        route_groups = lookup[route_experts]
        selected_positions = np.flatnonzero(route_groups >= 0)
        selected_groups = route_groups[selected_positions]
        order = np.argsort(selected_groups, kind="stable")
        sorted_positions = selected_positions[order]
        sorted_groups = selected_groups[order]
        observed_counts = tuple(
            int(value)
            for value in np.bincount(
                sorted_groups,
                minlength=len(active_experts),
            )
        )
        if observed_counts != counts:
            raise AssertionError(
                f"compact route counts {observed_counts} do not match {counts}"
            )
        token_sorted = sorted_positions // top_k
        unique_tokens, compact_route_tokens = np.unique(
            token_sorted,
            return_inverse=True,
        )
        compact_tokens = int(unique_tokens.size)
        if routed_pairs != int(sorted_positions.size):
            raise AssertionError(
                f"compact route plan contains {sorted_positions.size} routes, "
                f"expected {routed_pairs}"
            )
        route_ms = (
            (time.perf_counter_ns() - route_started) / 1e6 if profile else 0.0
        )

        input_started = time.perf_counter_ns()
        use_pinned_workspace = store.pin_memory and torch.cuda.is_available()
        workspace, workspace_allocations = self._workspace(
            hidden_size=store.shape.hidden_size,
            dtype=store.dtype,
            pin_memory=use_pinned_workspace,
            tokens=compact_tokens,
            routes=routed_pairs,
            scope=workspace_scope,
        )
        hidden_cpu = workspace.hidden[:compact_tokens]
        output_cpu = workspace.output[:compact_tokens]
        route_weights_cpu = workspace.route_weights[:routed_pairs]
        workspace_ready = workspace.output_reusable_event
        unique_tokens_array = np.ascontiguousarray(unique_tokens, dtype=np.int64)
        unique_tokens_source = torch.from_numpy(unique_tokens_array)
        if workspace.route_experts.is_pinned():
            unique_tokens_host = workspace.route_experts[:compact_tokens]
            unique_tokens_host.copy_(unique_tokens_source)
        else:
            unique_tokens_host = unique_tokens_source

        input_ready = None
        input_timing = None
        input_sources: tuple[torch.Tensor, ...] = ()
        if hidden_cpu.is_pinned() and hidden_states.device.type == "cuda":
            input_stream = self._input_stream(hidden_states.device)
            current_stream = torch.cuda.current_stream(hidden_states.device)
            input_stream.wait_stream(current_stream)
            input_start = torch.cuda.Event(enable_timing=profile)
            input_end = torch.cuda.Event(enable_timing=profile)
            with torch.cuda.stream(input_stream):
                input_start.record(input_stream)
                device_token_ids = unique_tokens_host.to(
                    device=hidden_states.device,
                    non_blocking=True,
                )
                compact_hidden = hidden_states.index_select(0, device_token_ids).to(
                    dtype=store.dtype
                )
                hidden_cpu.copy_(compact_hidden, non_blocking=True)
                input_end.record(input_stream)
            hidden_states.record_stream(input_stream)
            input_ready = input_end
            input_timing = (input_start, input_end) if profile else None
            input_sources = (compact_hidden,)
        else:
            device_token_ids = unique_tokens_host.to(device=hidden_states.device)
            compact_hidden = hidden_states.index_select(0, device_token_ids).to(
                dtype=store.dtype
            )
            hidden_cpu.copy_(compact_hidden)
        selected_weights = np.ascontiguousarray(
            route_packet_host[sorted_positions, 2],
            dtype=np.float32,
        )
        route_weights_cpu.copy_(
            torch.from_numpy(selected_weights).to(dtype=store.dtype)
        )
        input_copy_ms = (
            (time.perf_counter_ns() - input_started) / 1e6
            if profile and input_ready is None
            else 0.0
        )

        compact_route_tensor = torch.from_numpy(
            np.ascontiguousarray(compact_route_tokens, dtype=np.int64)
        )
        offsets: list[int] = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        jobs = []
        for group_id, expert_id in enumerate(active_experts):
            start = offsets[group_id]
            end = offsets[group_id + 1]
            jobs.append(
                _ExpertJob(
                    expert_id=expert_id,
                    source_tokens=compact_route_tensor[start:end],
                    output_tokens=compact_route_tensor[start:end],
                    weights=route_weights_cpu[start:end],
                    hidden=hidden_cpu,
                    store=store,
                    profile=profile,
                )
            )

        input_bytes = (
            hidden_cpu.numel() * hidden_cpu.element_size()
            + route_weights_cpu.numel() * route_weights_cpu.element_size()
        )
        output_bytes = output_cpu.numel() * output_cpu.element_size()
        return CPUPreparedExecution(
            jobs=tuple(jobs),
            output_cpu=output_cpu,
            output_tokens=unique_tokens_host,
            output_device=hidden_states.device,
            output_dtype=hidden_states.dtype,
            active_experts=len(active_experts),
            routed_pairs=routed_pairs,
            compact_tokens=compact_tokens,
            single_token_experts=sum(count == 1 for count in counts),
            small_experts=sum(count <= 4 for count in counts),
            max_tokens_per_expert=max(counts),
            workspace_allocations=workspace_allocations,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            input_copy_ms=input_copy_ms,
            route_ms=route_ms,
            profile=profile,
            device_token_ids=device_token_ids,
            workspace=workspace,
            workspace_ready=workspace_ready,
            input_ready=input_ready,
            input_timing=input_timing,
            input_sources=input_sources,
        )

    @torch.no_grad()
    def execute_prepared(self, prepared: CPUPreparedExecution) -> CPUExecutionResult:
        if prepared.workspace_ready is not None:
            prepared.workspace_ready.synchronize()
        input_copy_ms = prepared.input_copy_ms
        if prepared.input_ready is not None:
            prepared.input_ready.synchronize()
            if prepared.input_timing is not None:
                input_copy_ms = prepared.input_timing[0].elapsed_time(
                    prepared.input_timing[1]
                )
        prepared.output_cpu.zero_()
        compute_started = time.perf_counter_ns()
        if self._pool is None:
            expert_results = map(self._compute_expert, prepared.jobs)
        else:
            expert_results = self._pool.map(self._compute_expert, prepared.jobs)
        results = list(expert_results)

        gate_up_ms = sum(result.gate_up_ms for result in results)
        activation_ms = sum(result.activation_ms for result in results)
        down_ms = sum(result.down_ms for result in results)
        scatter_started = time.perf_counter_ns()
        for result in results:
            prepared.output_cpu.index_add_(0, result.output_tokens, result.output)
        scatter_ms = (
            (time.perf_counter_ns() - scatter_started) / 1e6 if prepared.profile else 0.0
        )
        compute_ms = (
            (time.perf_counter_ns() - compute_started) / 1e6 if prepared.profile else 0.0
        )
        return CPUExecutionResult(
            output=prepared.output_cpu,
            token_ids=prepared.output_tokens,
            active_experts=prepared.active_experts,
            routed_pairs=prepared.routed_pairs,
            compact_tokens=prepared.compact_tokens,
            single_token_experts=prepared.single_token_experts,
            small_experts=prepared.small_experts,
            max_tokens_per_expert=prepared.max_tokens_per_expert,
            workspace_allocations=prepared.workspace_allocations,
            input_bytes=prepared.input_bytes,
            output_bytes=prepared.output_bytes,
            input_copy_ms=input_copy_ms,
            route_ms=prepared.route_ms,
            compute_ms=compute_ms,
            gate_up_ms=gate_up_ms,
            activation_ms=activation_ms,
            down_ms=down_ms,
            scatter_ms=scatter_ms,
            output_copy_ms=0.0,
        )

    @torch.no_grad()
    def complete(
        self,
        prepared: CPUPreparedExecution,
        result: CPUExecutionResult,
    ) -> CPUExecutionResult:
        output_started = time.perf_counter_ns()
        output_token_ids = prepared.device_token_ids
        use_async_output = (
            output_token_ids is not None
            and result.output.is_pinned()
            and prepared.output_device.type == "cuda"
        )
        if use_async_output:
            result_stream = self._result_stream(prepared.output_device)
            current_stream = torch.cuda.current_stream(prepared.output_device)
            output_start = torch.cuda.Event(enable_timing=prepared.profile)
            output_ready = torch.cuda.Event(enable_timing=prepared.profile)
            with torch.cuda.stream(result_stream):
                output = torch.empty(
                    result.output.shape,
                    device=prepared.output_device,
                    dtype=prepared.output_dtype,
                )
                output_start.record(result_stream)
                output.copy_(result.output, non_blocking=True)
                output_ready.record(result_stream)
            current_stream.wait_event(output_ready)
            output.record_stream(current_stream)
            output_token_ids.record_stream(current_stream)
            if prepared.workspace is None:
                raise AssertionError("compact CPU output lost its workspace lease")
            prepared.workspace.output_reusable_event = output_ready
            if prepared.profile:
                output_ready.synchronize()
                output_copy_ms = output_start.elapsed_time(output_ready)
            else:
                output_copy_ms = 0.0
        else:
            output = result.output.to(
                device=prepared.output_device,
                dtype=prepared.output_dtype,
                copy=True,
            )
            if output_token_ids is None:
                output_token_ids = result.token_ids.to(
                    device=prepared.output_device,
                    copy=True,
                )
            output_copy_ms = (
                (time.perf_counter_ns() - output_started) / 1e6
                if prepared.profile
                else 0.0
            )
        return replace(
            result,
            output=output,
            token_ids=output_token_ids,
            output_copy_ms=output_copy_ms,
        )

    @torch.no_grad()
    def execute(
        self,
        *,
        store: LayerExpertStore,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        gate_weights: torch.Tensor,
        active_experts: tuple[int, ...],
        route_counts: dict[int, int],
        profile: bool,
    ) -> CPUExecutionResult:
        prepared = self.prepare(
            store=store,
            hidden_states=hidden_states,
            expert_ids=expert_ids,
            gate_weights=gate_weights,
            active_experts=active_experts,
            route_counts=route_counts,
            profile=profile,
        )
        return self.complete(prepared, self.execute_prepared(prepared))
