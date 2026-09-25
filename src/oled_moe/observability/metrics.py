from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class LayerMetrics:
    forward_calls: int = 0
    routed_pairs: int = 0
    cache_hit_pairs: int = 0
    cache_miss_pairs: int = 0
    active_experts: int = 0
    cache_hit_experts: int = 0
    cache_slot_observations: int = 0
    cache_occupied_slot_observations: int = 0
    priority_opportunity_slots: int = 0
    gpu_routed_pairs: int = 0
    gpu_chunk_executions: int = 0
    gpu_chunk_experts: int = 0
    gpu_chunk_padded_experts: int = 0
    cpu_routed_pairs: int = 0
    cpu_active_experts: int = 0
    cpu_compact_tokens: int = 0
    cpu_single_token_experts: int = 0
    cpu_small_experts: int = 0
    cpu_max_tokens_per_expert: int = 0
    cpu_workspace_allocations: int = 0
    cpu_dispatch_submissions: int = 0
    unique_miss_experts: int = 0
    transfer_priority_experts: int = 0
    cpu_priority_experts: int = 0
    transfer_priority_sum: float = 0.0
    cpu_priority_sum: float = 0.0
    h2d_expert_loads: int = 0
    transient_expert_loads: int = 0
    h2d_copy_submissions: int = 0
    h2d_bytes: int = 0
    retention_device_copies: int = 0
    retention_d2d_bytes: int = 0
    retention_capture_device_copies: int = 0
    retention_capture_d2d_bytes: int = 0
    priority_updates: int = 0
    priority_scheduled_calls: int = 0
    priority_eligible_calls: int = 0
    priority_bypassed_calls: int = 0
    route_accounting_calls: int = 0
    route_accounted_pairs: int = 0
    retention_updates: int = 0
    cpu_input_bytes: int = 0
    cpu_output_bytes: int = 0
    host_execute_ms: float = 0.0
    host_route_plan_ms: float = 0.0
    host_cache_stats_ms: float = 0.0
    host_priority_ms: float = 0.0
    host_dispatch_ms: float = 0.0
    host_cpu_input_ms: float = 0.0
    host_cpu_route_ms: float = 0.0
    host_cpu_compute_ms: float = 0.0
    host_cpu_gate_up_ms: float = 0.0
    host_cpu_activation_ms: float = 0.0
    host_cpu_down_ms: float = 0.0
    host_cpu_scatter_ms: float = 0.0
    host_cpu_output_ms: float = 0.0
    host_cpu_wait_ms: float = 0.0
    cuda_h2d_ms: float = 0.0
    cuda_transfer_wait_ms: float = 0.0
    cuda_retention_wait_ms: float = 0.0
    cuda_retention_d2d_ms: float = 0.0
    cuda_priority_ms: float = 0.0
    cuda_moe_compute_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.routed_pairs == 0:
            return 0.0
        return self.cache_hit_pairs / self.routed_pairs

    @property
    def expert_hit_rate(self) -> float:
        if self.active_experts == 0:
            return 0.0
        return self.cache_hit_experts / self.active_experts

    @property
    def cache_utilization(self) -> float:
        if self.cache_slot_observations == 0:
            return 0.0
        return self.cache_hit_experts / self.cache_slot_observations

    @property
    def cache_occupancy(self) -> float:
        if self.cache_slot_observations == 0:
            return 0.0
        return self.cache_occupied_slot_observations / self.cache_slot_observations

    @property
    def priority_opportunity(self) -> float:
        if self.cache_slot_observations == 0:
            return 0.0
        return self.priority_opportunity_slots / self.cache_slot_observations

    @property
    def priority_update_rate(self) -> float:
        if self.forward_calls == 0:
            return 0.0
        return self.priority_updates / self.forward_calls

    @property
    def transfer_mean_priority(self) -> float:
        if self.transfer_priority_experts == 0:
            return 0.0
        return self.transfer_priority_sum / self.transfer_priority_experts

    @property
    def cpu_mean_priority(self) -> float:
        if self.cpu_priority_experts == 0:
            return 0.0
        return self.cpu_priority_sum / self.cpu_priority_experts

    @property
    def h2d_effective_gbps(self) -> float:
        if self.cuda_h2d_ms == 0:
            return 0.0
        return self.h2d_bytes / self.cuda_h2d_ms / 1_000_000

    def as_dict(self) -> dict[str, int | float]:
        values = asdict(self)
        values["hit_rate"] = self.hit_rate
        values["expert_hit_rate"] = self.expert_hit_rate
        values["cache_utilization"] = self.cache_utilization
        values["cache_occupancy"] = self.cache_occupancy
        values["priority_opportunity"] = self.priority_opportunity
        values["priority_update_rate"] = self.priority_update_rate
        values["transfer_mean_priority"] = self.transfer_mean_priority
        values["cpu_mean_priority"] = self.cpu_mean_priority
        values["h2d_effective_gbps"] = self.h2d_effective_gbps
        return values
