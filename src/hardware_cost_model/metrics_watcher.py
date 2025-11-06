import requests, threading, time

# Stores latest metrics snapshot for each model
model_metrics = {}

# Stores last cumulative values (for delta computation)
_prev_values = {}

def fetch_vllm_metrics(model_name, url):
    """Fetch metrics from one vLLM endpoint and update global storage."""
    try:
        r = requests.get(url, timeout=2)
        if r.status_code != 200:
            return
        lines = r.text.splitlines()
    except Exception:
        return

    # Parse numeric metrics
    curr = {}
    for line in lines:
        if not line.startswith("vllm:"):
            continue
        value = float(line.split()[-1])
        if line.startswith("vllm:num_requests_running"):
            curr["num_requests_running"] = value
        elif line.startswith("vllm:num_requests_waiting"):
            curr["num_requests_waiting"] = value
        elif line.startswith("vllm:kv_cache_usage_perc"):
            curr["kv_cache_usage_perc"] = value
        elif line.startswith("vllm:time_to_first_token_seconds_sum"):
            curr["ttft_sum"] = value
        elif line.startswith("vllm:time_to_first_token_seconds_count"):
            curr["ttft_count"] = value
        elif line.startswith("vllm:inter_token_latency_seconds_sum"):
            curr["itl_sum"] = value
        elif line.startswith("vllm:inter_token_latency_seconds_count"):
            curr["itl_count"] = value
        elif line.startswith("vllm:e2e_request_latency_seconds_sum"):
            curr["e2e_sum"] = value
        elif line.startswith("vllm:e2e_request_latency_seconds_count"):
            curr["e2e_count"] = value
   
  
    # Compute running averages from deltas
    prev = _prev_values.get(model_name, {})
    data = {}

    print("curr : ",curr)
    print("prv : ",prev)

    for prefix in ["ttft", "itl", "e2e"]:
        sum_key, cnt_key = f"{prefix}_sum", f"{prefix}_count"
        cur_sum, cur_cnt = curr.get(sum_key, 0), curr.get(cnt_key, 0)
        prev_sum, prev_cnt = prev.get(sum_key, 0), prev.get(cnt_key, 0)

        delta_sum = cur_sum - prev_sum
        delta_cnt = cur_cnt - prev_cnt
        avg = (delta_sum / delta_cnt) if delta_cnt > 0 else 0.0

        data[f"{prefix}_avg"] = avg
        data[sum_key] = cur_sum
        data[cnt_key] = cur_cnt

    
    data["num_requests_running"] = curr.get("num_requests_running", 0)
    data["num_requests_waiting"] = curr.get("num_requests_waiting", 0)
    data["kv_cache_usage_perc"] = curr.get("kv_cache_usage_perc", 0)

    # Update global dicts
    _prev_values[model_name] = curr
    model_metrics[model_name] = data


def background_metrics_collector(model_url_map, interval=5):
    while True:
        for name, url in model_url_map.items():
            fetch_vllm_metrics(name, url)
        time.sleep(interval)


def start_metrics_watcher(model_url_map, interval=5):
    """Start background metrics watcher thread."""
    t = threading.Thread(target=background_metrics_collector,
                         args=(model_url_map, interval),
                         daemon=True)
    t.start()
    return t
