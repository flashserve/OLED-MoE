import json

from oled_moe.observability import write_performance_report


def test_performance_report_writes_json_and_html(tmp_path) -> None:
    layer = {
        "forward_calls": 2,
        "hit_rate": 0.75,
        "expert_hit_rate": 0.5,
        "cache_utilization": 0.25,
        "h2d_expert_loads": 3,
        "h2d_copy_submissions": 3,
        "h2d_bytes": 6_000_000,
        "cuda_h2d_ms": 1.0,
        "h2d_effective_gbps": 6.0,
        "cuda_transfer_wait_ms": 0.5,
        "cuda_moe_compute_ms": 2.0,
        "host_execute_ms": 3.0,
    }
    runtime = {
        "cache_hit_rate": 0.75,
        "expert_cache_hit_rate": 0.5,
        "cache_utilization": 0.25,
        "h2d_expert_loads": 3,
        "h2d_copy_submissions": 3,
        "h2d_bytes": 6_000_000,
        "h2d_effective_gbps": 6.0,
        "cuda_h2d_ms": 1.0,
        "cuda_transfer_wait_ms": 0.5,
        "cuda_moe_compute_ms": 2.0,
        "host_execute_ms": 3.0,
    }
    payload = {
        "created_at": "2026-08-30T12:00:00+08:00",
        "model": "/models/test",
        "configuration": {"cache_size": 4},
        "hardware": {"device_name": "test-gpu"},
        "benchmark": {
            "mean_latency_ms_per_output_token": 10.0,
            "mean_tps": 100.0,
            "mean_fps": 50.0,
            "peak_gpu_allocated_gib": 1.0,
            "runtime": runtime,
        },
        "layers": {1: layer},
    }

    json_path, html_path = write_performance_report(payload, tmp_path)

    assert json.loads(json_path.read_text())["model"] == "/models/test"
    html = html_path.read_text()
    assert "OLED-MoE Performance Dashboard" in html
    assert "test-gpu" in html
    assert "75.00%" in html
    assert "50.00%" in html
    assert "25.00%" in html
    assert "3 copy submissions" in html
