import httpx, time, csv, json, os, uuid, statistics, datetime
from typing import List

SERVER_URL = os.environ.get("VLLM_SERVER", "http://localhost:8001")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
INPUT_FILE = os.environ.get("PROMPTS_FILE", None)  # optional: path to txt with one prompt/line

# ---- Output paths (relative to src/router/) ----
OUT_CSV = "../../data/metrics_qwen3b.csv"
RUN_MANIFEST = "../../data/run_manifest.jsonl"
SNAPSHOT_DIR = "../../data/metrics_snapshots"

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
os.makedirs(os.path.dirname(RUN_MANIFEST), exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ---- Prompts ----
default_prompts = [
    "Explain quicksort in one sentence.",
    "Why is the sky blue?",
    "Write a short Python function to reverse a list.",
    "Summarize the theory of relativity in 2 lines.",
    "What is a GPU warp?",
    "What is the difference between prefill and decode in LLM inference?",
]

def load_prompts(path: str) -> List[str]:
    if not path:
        return default_prompts
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]

def get_models_max_len():
    try:
        with httpx.Client(timeout=10.0) as c:
            r = c.get(f"{SERVER_URL}/v1/models")
            data = r.json()
            for m in data.get("data", []):
                if m.get("id") == MODEL:
                    return m.get("max_model_len", None)
    except Exception:
        pass
    return None

def scrape_metrics_snapshot(req_id: str):
    # Save raw Prometheus text; parse later offline to avoid guessing names
    try:
        with httpx.Client(timeout=5.0) as c:
            r = c.get(f"{SERVER_URL}/metrics")
            text = r.text
        snap_path = os.path.join(SNAPSHOT_DIR, f"{req_id}.prom")
        with open(snap_path, "w") as f:
            f.write(text)
        return snap_path
    except Exception:
        return None

def main():
    prompts = load_prompts(INPUT_FILE)
    max_len = get_models_max_len()

    # write a tiny run manifest line (append)
    manifest = {
        "ts_iso": datetime.datetime.utcnow().isoformat() + "Z",
        "server_url": SERVER_URL,
        "model": MODEL,
        "num_prompts": len(prompts),
        "max_model_len": max_len,
        "gen_params": {"max_tokens": 128, "temperature": 0.7},
    }
    with open(RUN_MANIFEST, "a") as mf:
        mf.write(json.dumps(manifest) + "\n")

    rows = []
    for prompt in prompts:
        req_id = str(uuid.uuid4())
        ts_iso = datetime.datetime.utcnow().isoformat() + "Z"

        payload = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.7,
        }

        t0 = time.perf_counter()
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{SERVER_URL}/v1/completions", json=payload)
        t1 = time.perf_counter()

        total_s = t1 - t0
        data = resp.json()

        # tokens (usage block if available)
        if "usage" in data:
            ptoks = int(data["usage"].get("prompt_tokens", 0))
            otoks = int(data["usage"].get("completion_tokens", 0))
        else:
            # fallback approximations
            ptoks = len(prompt.split())
            otoks = len(data["choices"][0]["text"].split())

        # TTFT (server timing if exposed; else simple heuristic)
        ttft = None
        timing = data.get("timing", {})
        # vLLM often exposes fields like "time_to_first_token"
        if isinstance(timing, dict):
            ttft = timing.get("time_to_first_token", None)
        if ttft is None:
            # fallback heuristic: assume 20% of total is prefill when output is long
            ttft = total_s * 0.2 if otoks >= 64 else total_s * 0.35

        tpot_s = (total_s - ttft) / max(otoks, 1)

        # throughput estimates
        rpf = (ptoks / ttft) if ttft > 0 else None
        rdec = (1.0 / tpot_s) if tpot_s > 0 else None

        # metrics snapshot
        snap_path = scrape_metrics_snapshot(req_id)

        # one row
        rows.append([
            ts_iso, req_id, SERVER_URL, MODEL, max_len,
            ptoks, otoks,
            round(ttft, 6), round(tpot_s, 6), round(total_s, 6),
            round(rpf, 3) if rpf else "", round(rdec, 3) if rdec else "",
            json.dumps(payload, separators=(",", ":"))
        ])

    # write header if file new
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "ts_iso","req_id","server_url","model","max_model_len",
                "prompt_tokens","output_tokens",
                "ttft_s","tpot_s","total_s",
                "rpf_est_tok_per_s","rdec_est_tok_per_s",
                "gen_params_json"
            ])
        w.writerows(rows)

    # quick medians
    med_ttft = statistics.median([r[7] for r in rows])
    med_tpot = statistics.median([r[8] for r in rows])
    print(f"✅ Saved {len(rows)} rows to {OUT_CSV}")
    print(f"Median TTFT: {med_ttft:.3f}s | Median TPOT: {med_tpot:.4f}s/token")
    print(f"Raw Prometheus snapshots: {SNAPSHOT_DIR} (one .prom per request)")

if __name__ == "__main__":
    main()
