"""
build_hardware_cost_dataset.py
Collects per-request latency + hardware metrics from vLLM for cost-model training.
"""

import argparse, csv, os, random, time, uuid, yaml, requests, datetime, torch
from openai import OpenAI
from typing import List
from datasets import load_dataset

# ---------------------- CONFIG ----------------------

CSV_FIELDS = [
    # Request Metadata
    "request_id", "timestamp", "prompt_id", "model_id", "gpu_id",

    # Prompt Info
    "p_tokens",

    # Hardware Snapshot (Before Dispatch)
    "running_req_count", "waiting_req_count",
    "kv_cache_usage_perc", "gpu_free_mem_bytes",
    "prefill_tokens_per_s", "decode_tokens_per_s",

    # Latency Labels (After Completion)
    "ttft_s", "tpot_s_per_token", "latency_s",

    # Derived
    "slo_flag",

    # Output
    "d_tokens",
]



# ---------------- LOAD PROMPTS (MixInstruct) ----------------
def load_mix_instruct_prompts(n: int = 10, seed: int = 42):
    """
    Load Mix-Instruct validation split and return a list of
    (prompt_id, prompt_text) pairs for the first n prompts.
    """
    print(f"Loading Mix-Instruct (llm-blender/mix-instruct)...")
    ds = load_dataset("llm-blender/mix-instruct", split="validation")

    # Combine 'instruction' and 'input' like CSCR does
    def concat_prompt(x):
        inp = x["input"].strip() if x["input"] else ""
        return {"prompt": (x["instruction"].strip() + " " + inp).strip()}

    ds = ds.map(concat_prompt)

    # sample or take first n
    random.seed(seed)
    sampled = ds.select(range(min(n, len(ds))))

    # produce clean list of (id, prompt)
    prompts = [(row["id"], row["prompt"]) for row in sampled]
    print(f"Loaded {len(prompts)} prompts.")
    return prompts


# ---------------------- PROMETHEUS FETCHER ----------------------

def fetch_vllm_metrics(prom_url: str = "http://localhost:8000/metrics",
                       model_name: str = None,
                       gpu_id: int = 0):
    """Fetch only cost-relevant vLLM Prometheus metrics for the given GPU/model."""
    try:
        r = requests.get(prom_url, timeout=1)
        if r.status_code != 200:
            print(f"[WARN] Prometheus returned {r.status_code}")
            return {}
        lines = r.text.splitlines()
    except Exception as e:
        print(f"[WARN] Failed to fetch metrics: {e}")
        return {}

    metrics = {
        "running_req_count": 0.0,
        "waiting_req_count": 0.0,
        "kv_cache_usage_perc": 0.0,
        "gpu_free_mem_bytes": 0.0,
        "prefill_tokens_per_s": 0.0,   # derived later
        "decode_tokens_per_s": 0.0     # derived later
    }

    def parse_metric(prefix: str, filt: str = None):
        for line in lines:
            if line.startswith(prefix):
                if filt is None or filt in line:
                    try:
                        return float(line.split()[-1])
                    except Exception:
                        pass
        return 0.0

    # --- exact keys confirmed from your metrics dump ---
    metrics["running_req_count"] = parse_metric("vllm:num_requests_running", f'model_name="{model_name}"')
    metrics["waiting_req_count"] = parse_metric("vllm:num_requests_waiting", f'model_name="{model_name}"')
    metrics["kv_cache_usage_perc"] = parse_metric("vllm:kv_cache_usage_perc", f'model_name="{model_name}"')
    metrics["gpu_free_mem_bytes"] = parse_metric("vllm:gpu:memory_free_bytes", f'gpu_id="{gpu_id}"')

    # optional throughput (can compute externally)
    prompt_total = parse_metric("vllm:prompt_tokens_total", f'model_name="{model_name}"')
    gen_total = parse_metric("vllm:generation_tokens_total", f'model_name="{model_name}"')
    metrics["prefill_tokens_per_s"] = prompt_total
    metrics["decode_tokens_per_s"] = gen_total

    return metrics


import os

def dump_vllm_metrics_snapshot(prom_url: str = "http://localhost:8000/metrics",
                               dump_dir: str = "data/metrics_snapshots",
                               model_name: str = None):
    """Save one complete Prometheus /metrics snapshot to a timestamped text file."""
    import requests, datetime
    os.makedirs(dump_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_path = os.path.join(
        dump_dir,
        f"metrics_{model_name.replace('/', '_') if model_name else 'unknown'}_{timestamp}.txt"
    )

    try:
        r = requests.get(prom_url, timeout=2)
        if r.status_code == 200:
            with open(dump_path, "w") as f:
                f.write(r.text)
            print(f"[INFO] Saved vLLM metrics snapshot → {dump_path}")
        else:
            print(f"[WARN] Prometheus returned {r.status_code}, skipping dump.")
    except Exception as e:
        print(f"[WARN] Could not fetch metrics snapshot: {e}")

    return dump_path


# ---------------------- REQUEST HANDLER ----------------------

def send_request_and_measure(openai_client, model_name, prompt):
    """Send a single completion request and record TTFT, TPOT, latency."""
    start = time.time()
    request_id = str(uuid.uuid4())

    # --- send request
    stream = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=256,
    )

    first_token_time, total_tokens = None, 0
    for chunk in stream:
        if hasattr(chunk, "choices"):
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", "")
            if text:
                total_tokens += len(text.split())
                if first_token_time is None:
                    first_token_time = time.time()


    end = time.time()

    ttft = (first_token_time - start) if first_token_time else None
    latency = end - start
    tpot = (latency - ttft) / max(total_tokens, 1) if ttft else None

    return {
        "ttft_s": ttft or 0,
        "tpot_s_per_token": tpot or 0,
        "latency_s": latency,
        "d_tokens": total_tokens,
    }

# ---------------------- MAIN LOOP ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to GPU–Model YAML map.")
    parser.add_argument("--prom_url", default="http://localhost:8000/metrics")
    parser.add_argument("--output", default="data/hw_dataset.csv")
    parser.add_argument("--num_prompts", type=int, default=5)
    args = parser.parse_args()

    # --- load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    gpu_models = [(int(g), m["name"]) for g, lst in cfg["gpus"].items() for m in lst]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    prompts = load_mix_instruct_prompts(args.num_prompts)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for prompt_id, prompt in prompts:
            gpu_id, model_name = random.choice(gpu_models)
            gpu_name = torch.cuda.get_device_name(gpu_id)

            p_tokens = len(prompt.split())

            # ---- fetch metrics before dispatch
            metrics = fetch_vllm_metrics(args.prom_url, model_name, gpu_id)
            dump_vllm_metrics_snapshot(args.prom_url, model_name)

            # ---- send request
            latency_info = send_request_and_measure(client, model_name, prompt)

            slo_flag = int(latency_info["latency_s"] > 0.1 * (p_tokens + latency_info["d_tokens"]))

            row = {
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat(),
                "prompt_id": prompt_id,
                "model_id": model_name,
                "gpu_id": gpu_name,
                "p_tokens": p_tokens,
                "d_tokens": latency_info["d_tokens"],
                **metrics,
                **latency_info,
                "slo_flag": slo_flag,
            }
            writer.writerow(row)
            print(f"[{prompt_id}] {model_name} on GPU{gpu_id}: {latency_info['latency_s']:.3f}s")

    print(f"\n✅ Data collection complete → {args.output}")

# ----------------------

if __name__ == "__main__":
    main()
