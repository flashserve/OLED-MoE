from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _float(value: Any) -> float:
    return float(value or 0.0)


def _card(label: str, value: str, detail: str = "") -> str:
    return (
        '<div class="card">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="detail">{html.escape(detail)}</div>'
        "</div>"
    )


def write_performance_report(payload: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Write machine-readable metrics and a dependency-free HTML dashboard."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    benchmark = payload["benchmark"]
    runtime = benchmark["runtime"]
    layers = sorted(
        ((int(layer_id), values) for layer_id, values in payload["layers"].items()),
        key=lambda item: item[0],
    )
    max_wait = max((_float(values["cuda_transfer_wait_ms"]) for _, values in layers), default=0.0)

    cards = "".join(
        [
            _card(
                "Mean latency",
                f"{_float(benchmark['mean_latency_ms_per_output_token']):.2f} ms",
                "per output token",
            ),
            _card(
                "Throughput",
                f"{_float(benchmark['mean_tps']):.2f} tok/s",
                f"{_float(benchmark['mean_fps']):.2f} forwards/s",
            ),
            _card(
                "Routed-pair hit rate",
                f"{100 * _float(runtime['cache_hit_rate']):.2f}%",
                f"{int(runtime['h2d_expert_loads']):,} expert loads",
            ),
            _card(
                "Expert hit rate",
                f"{100 * _float(runtime.get('expert_cache_hit_rate')):.2f}%",
                "pre-cached hits / active experts",
            ),
            _card(
                "Cache utilization",
                f"{100 * _float(runtime.get('cache_utilization')):.2f}%",
                "pre-cached hits / cache slots",
            ),
            _card(
                "Priority opportunity",
                f"{100 * _float(runtime.get('priority_opportunity')):.2f}%",
                f"{int(runtime.get('priority_enabled_layers', 0)):,} selected layers; "
                f"{int(runtime.get('priority_bypassed_calls', 0)):,} updates bypassed",
            ),
            _card(
                "H2D payload",
                f"{_float(runtime['h2d_bytes']) / 1024**3:.2f} GiB",
                f"{int(runtime['h2d_copy_submissions']):,} copy submissions",
            ),
            _card(
                "H2D DMA",
                f"{_float(runtime['cuda_h2d_ms']):.2f} ms",
                "sum of copy-stream intervals",
            ),
            _card(
                "Transfer stall",
                f"{_float(runtime['cuda_transfer_wait_ms']):.2f} ms",
                "compute-stream critical wait",
            ),
            _card(
                "GPU MoE work",
                f"{_float(runtime['cuda_moe_compute_ms']):.2f} ms",
                "sum of profiled GPU spans",
            ),
            _card(
                "GPU expert batches",
                f"{int(runtime.get('gpu_chunk_executions', 0)):,}",
                f"{int(runtime.get('gpu_chunk_experts', 0)):,} real + "
                f"{int(runtime.get('gpu_chunk_padded_experts', 0)):,} padded experts",
            ),
            _card(
                "CPU expert work",
                f"{_float(runtime.get('host_cpu_compute_ms')):.2f} ms",
                f"{int(runtime.get('cpu_routed_pairs', 0)):,} routed pairs",
            ),
            _card(
                "CPU tail wait",
                f"{_float(runtime.get('host_cpu_wait_ms')):.2f} ms",
                f"{int(runtime.get('cpu_dispatch_submissions', 0)):,} asynchronous submissions",
            ),
            _card(
                "CPU GEMM breakdown",
                f"{_float(runtime.get('host_cpu_gate_up_ms')):.2f} / {_float(runtime.get('host_cpu_down_ms')):.2f} ms",
                "summed expert work: gate+up / down",
            ),
            _card(
                "CPU route shape",
                f"{int(runtime.get('cpu_active_experts', 0)):,} expert groups",
                f"{int(runtime.get('cpu_small_experts', 0)):,} groups have at most 4 routes",
            ),
            _card(
                "CPU activation I/O",
                f"{_float(runtime.get('host_cpu_input_ms')) + _float(runtime.get('host_cpu_output_ms')):.2f} ms",
                f"{(_float(runtime.get('cpu_input_bytes')) + _float(runtime.get('cpu_output_bytes'))) / 1024**3:.2f} GiB",
            ),
            _card(
                "CPU route grouping",
                f"{_float(runtime.get('host_cpu_route_ms')):.2f} ms",
                f"{int(runtime.get('cpu_workspace_allocations', 0)):,} measured reallocations",
            ),
            _card(
                "Host in OLED",
                f"{_float(runtime['host_execute_ms']):.2f} ms",
                "includes intentional GPU sync points",
            ),
            _card(
                "Priority overhead",
                f"{_float(runtime.get('cuda_priority_ms')):.2f} ms CUDA",
                f"{int(runtime.get('priority_updates', 0)):,} updates; "
                f"interval {int(runtime.get('priority_update_interval', 0))}; "
                f"{100 * _float(runtime.get('priority_update_rate')):.1f}% layer calls; "
                f"{_float(runtime.get('host_priority_ms')):.2f} ms host path; "
                f"{int(runtime.get('retention_capture_device_copies', 0)):,} capture + "
                f"{int(runtime.get('retention_device_copies', 0)):,} restore copies",
            ),
            _card(
                "Route accounting",
                f"{int(runtime.get('route_accounting_calls', 0)):,} host handoffs",
                f"{int(runtime.get('route_accounted_pairs', 0)):,} token-expert routes; "
                f"{int(runtime.get('retention_updates', 0)):,} retention updates",
            ),
            _card(
                "Layer capacity",
                f"{int(runtime.get('layer_capacity_recommended_min', 0))}–{int(runtime.get('layer_capacity_recommended_max', 0))} slots",
                f"{'applied' if runtime.get('layer_capacity_apply') else 'dry-run'}; "
                f"{int(runtime.get('layer_capacity_updates', 0)):,} updates; "
                f"{int(runtime.get('layer_capacity_total_slots', 0)):,} total slots",
            ),
            _card(
                "Peak GPU allocated",
                f"{_float(benchmark['peak_gpu_allocated_gib']):.2f} GiB",
                "PyTorch allocator",
            ),
        ]
    )

    rows = []
    for layer_id, values in layers:
        wait_ms = _float(values["cuda_transfer_wait_ms"])
        bar_width = 0.0 if max_wait == 0 else min(100.0, 100.0 * wait_ms / max_wait)
        rows.append(
            "<tr>"
            f"<td>{layer_id}</td>"
            f'<td class="num">{int(values.get("layer_capacity_current", 0))}</td>'
            f'<td class="num">{int(values.get("layer_capacity_recommended", 0))}</td>'
            f'<td class="num">{100 * _float(values.get("layer_activation_ratio_ema")):.2f}%</td>'
            f'<td class="num">{100 * _float(values.get("layer_cache_utilization_ema")):.2f}%</td>'
            f'<td class="num">{_float(values.get("layer_capacity_score")):.4f}</td>'
            f'<td class="num">{int(values["forward_calls"]):,}</td>'
            f'<td class="num">{100 * _float(values.get("expert_hit_rate")):.2f}%</td>'
            f'<td class="num">{100 * _float(values.get("cache_utilization")):.2f}%</td>'
            f'<td class="num">{100 * _float(values.get("priority_opportunity")):.2f}%</td>'
            f'<td class="num">{100 * _float(values.get("priority_opportunity_ema")):.2f}%</td>'
            f'<td class="num">{int(values.get("priority_enabled", 0))}</td>'
            f'<td class="num">{int(values.get("priority_updates", 0)):,}</td>'
            f'<td class="num">{100 * _float(values["hit_rate"]):.2f}%</td>'
            f'<td class="num">{int(values.get("cpu_routed_pairs", 0)):,}</td>'
            f'<td class="num">{int(values.get("cpu_active_experts", 0)):,}</td>'
            f'<td class="num">{int(values.get("cpu_compact_tokens", 0)):,}</td>'
            f'<td class="num">{_float(values.get("host_cpu_compute_ms")):.2f}</td>'
            f'<td class="num">{_float(values.get("host_cpu_wait_ms")):.2f}</td>'
            f'<td class="num">{int(values["h2d_expert_loads"]):,}</td>'
            f'<td class="num">{int(values["h2d_copy_submissions"]):,}</td>'
            f'<td class="num">{_float(values["h2d_bytes"]) / 1024**3:.2f}</td>'
            f'<td class="num">{_float(values["cuda_h2d_ms"]):.2f}</td>'
            f'<td class="num">{_float(values["h2d_effective_gbps"]):.2f}</td>'
            f'<td class="num wait"><span>{wait_ms:.2f}</span><i style="width:{bar_width:.2f}%"></i></td>'
            f'<td class="num">{_float(values["cuda_moe_compute_ms"]):.2f}</td>'
            f'<td class="num">{_float(values["host_execute_ms"]):.2f}</td>'
            "</tr>"
        )

    measured_wall_ms = sum(_float(run.get("seconds")) for run in payload.get("runs", [])) * 1000
    dma_ms = _float(runtime["cuda_h2d_ms"])
    wait_ms = _float(runtime["cuda_transfer_wait_ms"])
    exposed_percent = 100 * wait_ms / dma_ms if dma_ms else 0.0
    wall_wait_percent = 100 * wait_ms / measured_wall_ms if measured_wall_ms else 0.0
    wall_moe_percent = (
        100 * _float(runtime["cuda_moe_compute_ms"]) / measured_wall_ms if measured_wall_ms else 0.0
    )
    top_wait_layers = ", ".join(
        str(layer_id)
        for layer_id, _ in sorted(
            layers,
            key=lambda item: _float(item[1]["cuda_transfer_wait_ms"]),
            reverse=True,
        )[:3]
    )
    observations = "".join(
        [
            f"<li><strong>{_float(runtime['h2d_effective_gbps']):.2f} GB/s</strong> effective H2D bandwidth across profiled copies.</li>",
            f"<li><strong>{int(runtime['h2d_copy_submissions']):,}</strong> H2D submissions serve {int(runtime['h2d_expert_loads']):,} expert loads.</li>",
            f"<li><strong>{exposed_percent:.1f}%</strong> of copy-stream DMA time remains visible as compute-stream waiting; lower is better.</li>",
            f"<li>Transfer waiting occupies <strong>{wall_wait_percent:.1f}%</strong> of measured end-to-end wall time; profiled GPU MoE work occupies {wall_moe_percent:.1f}%.</li>",
            f"<li>Layers <strong>{html.escape(top_wait_layers)}</strong> have the largest transfer stalls and are the first capacity/prefetch targets.</li>",
            "<li>Host routing retains one required D2H metadata handoff for the Python cache scheduler. Route counts share that handoff; the execution hot path has no additional `.item()` synchronization.</li>",
        ]
    )

    config_json = html.escape(json.dumps(payload["configuration"], indent=2, sort_keys=True))
    hardware_json = html.escape(json.dumps(payload["hardware"], indent=2, sort_keys=True))
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OLED-MoE Performance Dashboard</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#141b2d; --line:#29324a; --text:#e8edf7; --muted:#9eabc3; --accent:#67e8c3; --warn:#f7b955; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 Inter,system-ui,sans-serif; }}
main {{ max-width:1500px; margin:auto; padding:34px; }}
h1 {{ margin:0; font-size:30px; }}
.subtitle {{ color:var(--muted); margin:6px 0 26px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
.card,section {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; }}
.card {{ padding:16px; }} .label,.detail {{ color:var(--muted); }} .value {{ font-size:23px; font-weight:700; margin:5px 0; }}
section {{ margin-top:18px; padding:18px; overflow:auto; }} h2 {{ font-size:17px; margin:0 0 12px; }}
table {{ border-collapse:collapse; width:100%; min-width:1050px; }} th,td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:left; white-space:nowrap; }}
th {{ color:var(--muted); font-weight:600; }} td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
td.wait {{ position:relative; }} td.wait span {{ position:relative; z-index:1; }} td.wait i {{ position:absolute; left:0; bottom:3px; height:3px; background:var(--warn); opacity:.8; }}
.note {{ color:var(--muted); }} pre {{ color:#c7d2e8; margin:0; }} .split {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
@media(max-width:800px) {{ main {{ padding:18px; }} .split {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
<h1>OLED-MoE Performance Dashboard</h1>
<div class="subtitle">{html.escape(str(payload["created_at"]))} · {html.escape(str(payload["model"]))}</div>
<div class="cards">{cards}</div>
<section><h2>Automated observations</h2><ul>{observations}</ul></section>
<section><h2>Per-layer profile</h2>
<table><thead><tr><th>Layer</th><th>Current slots</th><th>Recommended slots</th><th>Activation EMA</th><th>Utilization EMA</th><th>Allocation score</th><th>Calls</th><th>Expert hit rate</th><th>Cache utilization</th><th>Priority opportunity</th><th>Opportunity EMA</th><th>Selected</th><th>Priority updates</th><th>Pair hit rate</th><th>CPU pairs</th><th>CPU groups</th><th>Compact tokens</th><th>CPU compute ms</th><th>CPU tail wait ms</th><th>H2D experts</th><th>Copy submissions</th><th>H2D GiB</th><th>DMA ms</th><th>GB/s</th><th>Wait ms</th><th>GPU MoE ms</th><th>Host ms</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p class="note">CUDA interval sums represent work on their respective streams and may overlap. Transfer stall is measured on the compute stream and is the more useful critical-path signal. Host time includes the required dynamic routing metadata handoff.</p>
</section>
<div class="split"><section><h2>Configuration</h2><pre>{config_json}</pre></section><section><h2>Hardware</h2><pre>{hardware_json}</pre></section></div>
</main></body></html>"""
    html_path = output_dir / "report.html"
    html_path.write_text(body, encoding="utf-8")
    return json_path, html_path
