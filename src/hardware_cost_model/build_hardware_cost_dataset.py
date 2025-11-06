"""
build_hardware_cost_dataset.py
Collects per-request latency + hardware metrics from vLLM for cost-model training.
"""

import argparse, csv, os, random, time, uuid, yaml, datetime, torch
from openai import OpenAI
from datasets import load_dataset
from .metrics_watcher import start_metrics_watcher, model_metrics

# ---------------------- CONFIG ----------------------

CSV_FIELDS = [
    # Request Metadata
    "request_id", "timestamp", "prompt_id", "model_id", "gpu_id",

    # Prompt Info
    "p_tokens",

    # Hardware Snapshot (Before Dispatch)
    "running_req_count", "waiting_req_count",
    "kv_cache_usage_perc",
    "ttft_avg", "itl_avg", "e2e_avg",

    # Latency Labels (After Completion)
    "ttft_s", "tpot_s_per_token", "latency_s",

    # Derived
    "slo_flag",

    # Output
    "d_tokens",
]


# ---------------------- PROMPTS ----------------------

def load_mix_instruct_prompts(n: int = 10, seed: int = 42):
    """Load Mix-Instruct validation split and return (prompt_id, prompt_text) pairs."""
    print(f"Loading Mix-Instruct (llm-blender/mix-instruct)...")
    ds = load_dataset("llm-blender/mix-instruct", split="validation")

    def concat_prompt(x):
        inp = x["input"].strip() if x["input"] else ""
        return {"prompt": (x["instruction"].strip() + " " + inp).strip()}

    ds = ds.map(concat_prompt)
    random.seed(seed)
    sampled = ds.select(range(min(n, len(ds))))
    prompts = [(row["id"], row["prompt"]) for row in sampled]
    print(f"Loaded {len(prompts)} prompts.")
    return prompts


# ---------------------- REQUEST HANDLER ----------------------

def send_request_and_measure(openai_client, model_name, prompt):
    """Send a single completion request and record TTFT, TPOT, latency."""
    start = time.time()
    request_id = str(uuid.uuid4())

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
    ttft = (first_token_time - start) if first_token_time else 0
    latency = end - start
    tpot = (latency - ttft) / max(total_tokens, 1) if ttft > 0 else 0

    return {
        "ttft_s": ttft,
        "tpot_s_per_token": tpot,
        "latency_s": latency,
        "d_tokens": total_tokens,
    }


# ---------------------- MAIN ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to GPU–Model YAML map.")
    parser.add_argument("--output", default="data/hw_dataset.csv")
    parser.add_argument("--num_prompts", type=int, default=5)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()

    # --- Load config (YAML: GPU → [models])
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    gpu_models = [(int(g), m["name"], m["url"]) for g, lst in cfg["gpus"].items() for m in lst]

    # Build model→URL map for watcher
    model_url_map = {m["name"]: m["url"] for g, lst in cfg["gpus"].items() for m in lst}

    # --- Start metrics watcher
    print("Starting metrics watcher...")
    start_metrics_watcher(model_url_map, interval=args.interval)

    # --- Prepare output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    clients = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY"),
        "Qwen/Qwen1.5-0.5B": OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY"),
    }

    prompts = load_mix_instruct_prompts(args.num_prompts)

    # --- Collect data
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for prompt_id, prompt in prompts:
            gpu_id, model_name, _ = random.choice(gpu_models)
            gpu_name = torch.cuda.get_device_name(gpu_id)
            p_tokens = len(prompt.split())

            # Fetch latest hardware snapshot from watcher
            hw = model_metrics.get(model_name, {})
            metrics_snapshot = {
                "running_req_count": hw.get("num_requests_running", 0),
                "waiting_req_count": hw.get("num_requests_waiting", 0),
                "kv_cache_usage_perc": hw.get("kv_cache_usage_perc", 0),
                "ttft_avg": hw.get("ttft_avg", 0),
                "itl_avg": hw.get("itl_avg", 0),
                "e2e_avg": hw.get("e2e_avg", 0),
            }

            # Send request and record latency
            client = clients[model_name]
            latency_info = send_request_and_measure(client, model_name, prompt)

            slo_flag = int(
                latency_info["latency_s"] > 0.1 * (p_tokens + latency_info["d_tokens"])
            )

            row = {
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat(),
                "prompt_id": prompt_id,
                "model_id": model_name,
                "gpu_id": gpu_name,
                "p_tokens": p_tokens,
                "d_tokens": latency_info["d_tokens"],
                **metrics_snapshot,
                **latency_info,
                "slo_flag": slo_flag,
            }
            writer.writerow(row)
            print(f"[{prompt_id}] {model_name} on GPU{gpu_id}: {latency_info['latency_s']:.3f}s")

    print(f"\n✅ Data collection complete → {args.output}")


if __name__ == "__main__":
    main()
