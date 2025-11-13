# eval_pipeline.py

import os, csv, uuid, time, datetime, yaml, threading, random
import pandas as pd
from openai import OpenAI
import torch
from .routers_baseline import BaselineRouter, BaselineRouterWithOurCost
from .cost_model import HardwareCostPredictor
from .metrics_watcher import model_metrics
from .load_patterns import RequestPattern


# ======================
#   CSV SCHEMA
# ======================
CSV_FIELDS = [
    "run_id",
    "router_id",

    "pattern_type",
    "arrival_rate",

    "request_id",
    "timestamp",
    "prompt_id",

    "model_id",
    "gpu_id",

    "p_tokens",
    "d_tokens",

    "running_req_count",
    "waiting_req_count",
    "kv_cache_usage_perc",

    "pred_ttft",
    "pred_tpot",

    "observed_ttft",
    "observed_tpot",
    "latency_s",

    "slo_limit_s",
    "slo_met",
]


# ======================
#   Sending Request
# ======================
def send_request(client, model, prompt):
    start = time.time()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    first_token_time, ntoks = None, 0
    for ch in stream:
        if hasattr(ch, "choices"):
            txt = getattr(ch.choices[0].delta, "content", "")
            if txt:
                ntoks += len(txt.split())
                if first_token_time is None:
                    first_token_time = time.time()

    end = time.time()
    ttft = max((first_token_time - start), 0) if first_token_time else 0
    lat = end - start
    tpot = (lat - ttft) / max(ntoks, 1) if ttft > 0 else 0

    return ttft, tpot, lat, ntoks


# ======================
#   Main Evaluation
# ======================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--output", default="data/eval_results.csv")
    parser.add_argument("--num_prompts", type=int, default=50)
    parser.add_argument("--pattern", default="poisson")
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--router", choices=["baseline", "baseline+ourcost"], required=True)
    parser.add_argument("--model_path", default="checkpoints/hardware_cost_model/model.pt")
    parser.add_argument("--preproc_path", default="checkpoints/hardware_cost_model/preproc.joblib")
    args = parser.parse_args()

    RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"[Eval] Starting evaluation → {args.router}, run_id={RUN_ID}")

    # ---------- Load Config ----------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---------- Clients ----------
    clients, model_to_gpu = {}, {}
    for g, models in cfg["gpus"].items():
        gid = int(g)
        for m in models:
            nm = m["name"]
            base = m["base_url"].rstrip("/")
            clients[nm] = OpenAI(base_url=f"{base}/v1", api_key="EMPTY")
            model_to_gpu[nm] = gid

    # ---------- Prompts ----------
    df = pd.read_parquet(args.prompt_path)
    df = df.sample(n=args.num_prompts, random_state=42)
    prompts = [(str(i), row["prompt"]) for i, row in df.iterrows()]
    print(f"[Eval] Loaded {len(prompts)} prompts.")

    # ---------- Router ----------
    model_names = list(clients.keys())

    if args.router == "baseline":
        router = BaselineRouter(model_names)
        cost_predictor = None
    else:
        cost_predictor = HardwareCostPredictor(args.model_path, args.preproc_path)
        router = BaselineRouterWithOurCost(model_names, cost_predictor)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer_lock = threading.Lock()
    pattern = RequestPattern(args.pattern, args.rate)

    # ---------- SLO Function ----------
    A_SLO = 0.5
    B_SLO = 0.002

    def slo_limit(p, d):
        return A_SLO + B_SLO * (p + d)

    # ======================
    #   Worker
    # ======================
    def worker(pid, prompt):
        hw = model_metrics
        p_tokens = len(prompt.split())

        # feature snapshot (same design as training)
        m0 = random.choice(model_names)
        gpu0 = model_to_gpu[m0]
        snap = hw.get(m0, {})

        feat = {
            "p_tokens": p_tokens,
            "running_req_count": snap.get("num_requests_running", 0),
            "waiting_req_count": snap.get("num_requests_waiting", 0),
            "kv_cache_usage_perc": snap.get("kv_cache_usage_perc", 0),
            "model_id": m0,
            "gpu_id": gpu0,
        }

        # router decision
        if cost_predictor:
            selected = router.route(prompt, feat)
            pred_ttft, pred_tpot = cost_predictor(selected, feat)
        else:
            selected = router.route(prompt)
            pred_ttft = pred_tpot = 0

        gpu_id = model_to_gpu[selected]
        client = clients[selected]

        # actual execution
        obs_ttft, obs_tpot, lat, d_tokens = send_request(client, selected, prompt)

        # SLO
        limit = slo_limit(p_tokens, d_tokens)
        slo_met = int(lat <= limit)

        # write row
        row = {
            "run_id": RUN_ID,
            "router_id": args.router,

            "pattern_type": args.pattern,
            "arrival_rate": args.rate,

            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt_id": pid,

            "model_id": selected,
            "gpu_id": gpu_id,

            "p_tokens": p_tokens,
            "d_tokens": d_tokens,

            "running_req_count": feat["running_req_count"],
            "waiting_req_count": feat["waiting_req_count"],
            "kv_cache_usage_perc": feat["kv_cache_usage_perc"],

            "pred_ttft": pred_ttft,
            "pred_tpot": pred_tpot,

            "observed_ttft": obs_ttft,
            "observed_tpot": obs_tpot,
            "latency_s": lat,

            "slo_limit_s": limit,
            "slo_met": slo_met,
        }

        with writer_lock:
            file_exists = os.path.exists(args.output)
            with open(args.output, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if not file_exists:
                    w.writeheader()
                w.writerow(row)

    # ======================
    #   Launch Threads
    # ======================
    threads = []
    for i, (pid, prompt) in enumerate(prompts, start=1):

        if i % 20 == 0:
            print(f"[Eval] Progress: {i}/{len(prompts)}")

        while len([t for t in threads if t.is_alive()]) >= 10:
            time.sleep(0.02)

        t = threading.Thread(target=worker, args=(pid, prompt))
        threads.append(t)
        t.start()
        time.sleep(pattern.next_delay())

    for t in threads:
        t.join()

    print(f"[Eval] Done → results saved to {args.output}")


if __name__ == "__main__":
    main()
