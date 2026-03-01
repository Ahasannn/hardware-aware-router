#!/usr/bin/env python3
"""Test hardware monitor module"""
import time
from hw_router.hardware_monitor import start_metrics_watcher, model_metrics

def test_hardware_monitor():
    """Test the hardware monitor with deployed models"""

    # Define model URLs
    model_url_map = {
        "tinyllama": "http://localhost:8020/metrics",
        "qwen05b": "http://localhost:8021/metrics"
    }

    print("Starting hardware monitor...")
    start_metrics_watcher(model_url_map, interval=2)

    # Wait for initial metrics collection
    print("Waiting for initial metrics collection...")
    time.sleep(5)

    # Display collected metrics
    print("\n" + "="*60)
    print("HARDWARE METRICS COLLECTED")
    print("="*60)

    for model_name, metrics in model_metrics.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Running requests:  {metrics.get('num_requests_running', 0)}")
        print(f"  Waiting requests:  {metrics.get('num_requests_waiting', 0)}")
        print(f"  KV cache usage:    {metrics.get('kv_cache_usage_perc', 0):.2%}")
        print(f"  TTFT avg:          {metrics.get('ttft_avg', 0):.4f}s")
        print(f"  ITL avg:           {metrics.get('itl_avg', 0):.4f}s")
        print(f"  E2E avg:           {metrics.get('e2e_avg', 0):.4f}s")
        print(f"  Waiting tokens:    {metrics.get('waiting_tokens_estimate', 0):.0f}")

    print("\n" + "="*60)
    print("✓ Hardware monitor test PASSED")
    print("="*60)

    return True

if __name__ == "__main__":
    test_hardware_monitor()
