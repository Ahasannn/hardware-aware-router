"""
build_hardware_cost_dataset.py
Collects per-request latency + hardware metrics from vLLM for cost-model training.
Supports concurrent requests to create GPU contention.
"""

import argparse, csv, os, random, time, uuid, yaml, datetime, torch, threading
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
    "running_req_count", "waiting_req_count", "kv_cache_usage_perc",
    "ttft_avg", "itl_avg", "e2e_avg",
    # Latency Labels (After Completion)
    "ttft_s", "tpot_s_per_token", "latency_s",
    # Output
    "d_tokens",
]


# ---------------------- PROMPTS ----------------------

def load_mix_instruct_prompts(n: int = 10):
    """Load first n prompts from Mix-Instruct training split."""
    print("Loading Mix-Instruct (llm-blender/mix-instruct, split='train')...")
    ds = load_dataset("llm-blender/mix-instruct", split="train")

    # Combine instruction + input fields into a single prompt
    def concat_prompt(x):
        inp = x["input"].strip() if x["input"] else ""
        return {"prompt": (x["instruction"].strip() + " " + inp).strip()}

    ds = ds.map(concat_prompt)

    # Select the first n examples directly (serial order)
    sampled = ds.select(range(min(n, len(ds))))

    # Build (id, prompt) list
    prompts = [(row["id"], row["prompt"]) for row in sampled]

    print(f"Loaded {len(prompts)} prompts (first {n} from training split).")
    return prompts



# ---------------------- REQUEST HANDLER ----------------------

def send_request_and_measure(openai_client, model_name, prompt):
    """Send a single completion request and record TTFT, TPOT, latency."""
    start = time.time()

    stream = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=1024,
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


# ---------------------- WORKER FUNCTION ----------------------

def handle_request(prompt_id, prompt, gpu_models, clients, args, writer_lock):
    """Single threaded worker that sends one request and logs result."""
    gpu_id, model_name, _ = random.choice(gpu_models)
    gpu_name = torch.cuda.get_device_name(gpu_id)
    p_tokens = len(prompt.split())
    client = clients[model_name]

    # Fetch current hardware snapshot
    hw = model_metrics.get(model_name, {})
    metrics_snapshot = {
        "running_req_count": hw.get("num_requests_running", 0),
        "waiting_req_count": hw.get("num_requests_waiting", 0),
        "kv_cache_usage_perc": hw.get("kv_cache_usage_perc", 0),
        "ttft_avg": hw.get("ttft_avg", 0),
        "itl_avg": hw.get("itl_avg", 0),
        "e2e_avg": hw.get("e2e_avg", 0),
    }

    # Send request
    latency_info = send_request_and_measure(client, model_name, prompt)

    row = {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt_id": prompt_id,
        "model_id": model_name,
        "gpu_id": gpu_name,
        "p_tokens": p_tokens,
        "d_tokens": latency_info["d_tokens"],
        **metrics_snapshot,
        **latency_info
    }

    # Thread-safe file write
    with writer_lock:
        file_exists = os.path.exists(args.output)
        with open(args.output, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"[{prompt_id}] {model_name} latency={latency_info['latency_s']:.3f}s")


# ---------------------- MAIN ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to GPU–Model YAML map.")
    parser.add_argument("--output", default="data/hw_dataset.csv")
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent threads")
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

    # --- Prepare clients per model
    clients = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY"),
        "Qwen/Qwen1.5-0.5B": OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY"),
    }

    # --- Load prompts
    prompts = load_mix_instruct_prompts(args.num_prompts)

    # --- Create threads to simulate concurrent load
    threads, writer_lock = [], threading.Lock()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for prompt_id, prompt in prompts:
        t = threading.Thread(
            target=handle_request,
            args=(prompt_id, prompt, gpu_models, clients, args, writer_lock)
        )
        t.daemon = True
        t.start()
        time.sleep(random.uniform(0.0, 0.05))  # smaller stagger

    # Wait for all threads to complete
    for th in threads:
        th.join()

    print(f"\n✅ Data collection complete → {args.output}")


if __name__ == "__main__":
    main()
