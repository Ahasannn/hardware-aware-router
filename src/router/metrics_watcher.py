import requests, re

store_dict = {}
last_values = {}

def fetch_vllm_metrics(url):
    """Fetch vLLM metrics (multiple models on one GPU)."""
    try:
        r = requests.get(url, timeout=2)
        if r.status_code != 200:
            return None
    except Exception:
        return None

    lines = r.text.splitlines()

    for line in lines:
        if not line.startswith("vllm:"):
            continue

        # extract model name
        m = re.search(r'model_name="([^"]+)"', line)
        if not m:
            continue
        model = m.group(1)
        if model not in store_dict:
            store_dict[model] = {}
            last_values.setdefault(model, {})

        value = float(line.split()[-1])

        # ---- simple metrics ----
        if line.startswith("vllm:num_requests_running"):
            store_dict[model]["num_requests_running"] = value
        elif line.startswith("vllm:num_requests_waiting"):
            store_dict[model]["num_requests_waiting"] = value
        elif line.startswith("vllm:kv_cache_usage_perc"):
            store_dict[model]["kv_cache_usage_perc"] = value

        # ---- histogram metrics (count/sum) ----
        elif "time_to_first_token_seconds_count" in line:
            _update_hist_metric(model, "ttft_count", value)
        elif "time_to_first_token_seconds_sum" in line:
            _update_hist_metric(model, "ttft_sum", value)
        elif "inter_token_latency_seconds_count" in line:
            _update_hist_metric(model, "itl_count", value)
        elif "inter_token_latency_seconds_sum" in line:
            _update_hist_metric(model, "itl_sum", value)
        elif "e2e_request_latency_seconds_count" in line:
            _update_hist_metric(model, "e2e_count", value)
        elif "e2e_request_latency_seconds_sum" in line:
            _update_hist_metric(model, "e2e_sum", value)

    # ---- compute averages ----
    for m, d in store_dict.items():
        for prefix in ["ttft", "itl", "e2e"]:
            c, s = d.get(f"{prefix}_count", 0), d.get(f"{prefix}_sum", 0)
            d[f"{prefix}_avg"] = (s / c) if c else 0.0

    return store_dict


def _update_hist_metric(model, key, value):
    """Store deltas for running averages."""
    prev = last_values[model].get(key, 0)
    store_dict[model][key] = max(0, value - prev)
    last_values[model][key] = value
