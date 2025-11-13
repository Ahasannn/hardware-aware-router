# eval_pipeline.py

import os, csv, uuid, time, datetime, yaml, threading
import pandas as pd
from openai import OpenAI
from .routers import (
    BaselineRouter,
    RandomRouter,
    RoundRobinRouter,
    CarrotRouter
)
from .cost_model import HardwareCostPredictor
from .metrics_watcher import model_metrics
from .load_patterns import RequestPattern
from carrot import load_carrot_router


CSV_FIELDS = [
    "run_id", "router_id",
    "pattern_type", "arrival_rate",
    "request_id", "timestamp", "prompt_id",
    "model_id", "gpu_id",
    "p_tokens", "d_tokens",
    "running_req_count", "waiting_req_count", "kv_cache_usage_perc",
    "router_quality", "router_cost",
    "pred_ttft", "pred_tpot",
    "final_cost", "final_score",
    "observed_ttft", "observed_tpot", "latency_s",
    "slo_limit_s", "slo_met"
]


def send_request(client, model, prompt):
    start = time.time()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    first, ntoks = None, 0
    for ch in stream:
        if hasattr(ch, "choices"):
            txt = getattr(ch.choices[0].delta, "content", "")
            if txt:
                ntoks += len(txt.split())
                if first is None:
                    first = time.time()

    end = time.time()
    ttft = max((first - start), 0) if first else 0
    lat = end - start
    tpot = (lat - ttft) / max(ntoks, 1) if ttft > 0 else 0
    return ttft, tpot, lat, ntoks


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt_path", default="data/prompts/mixed_prompts_eval.parquet")
    parser.add_argument("--output", default="data/eval_results.csv")
    parser.add_argument("--router", choices=["baseline", "random", "rr", "carrot"], required=True)
    parser.add_argument("--use_hw_cost", action="store_true")
    parser.add_argument("--cost_lambda", type=float, default=1.0)
    parser.add_argument("--num_prompts", type=int, default=50)
    parser.add_argument("--pattern", default="poisson")
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--model_path", default="checkpoints/hardware_cost_model/model.pt")
    parser.add_argument("--preproc_path", default="checkpoints/hardware_cost_model/preproc.joblib")
    args = parser.parse_args()

    RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[Eval] Router={args.router}, use_hw_cost={args.use_hw_cost}, run_id={RUN_ID}")

    # ---------------------------------
    # Load router
    # ---------------------------------
    if args.router == "baseline":
        router = BaselineRouter()
    elif args.router == "random":
        router = RandomRouter()
    elif args.router == "rr":
        router = RoundRobinRouter()
    elif args.router == "carrot":
        carrot = load_carrot_router("checkpoints/carrot", model_type="linear")
        prices = {}  # TODO fill your price-per-model here
        router = CarrotRouter(carrot, prices)
    else:
        raise ValueError("Invalid router")

    # ---------------------------------
    # Cost model (only used if use_hw_cost=True)
    # ---------------------------------
    cost_predictor = None
    if args.use_hw_cost:
        cost_predictor = HardwareCostPredictor(args.model_path, args.preproc_path)

    # ---------------------------------
    # Load config (vLLM endpoints)
    # ---------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    clients, model_to_gpu = {}, {}
    for g, models in cfg["gpus"].items():
        gid = int(g)
        for m in models:
            nm = m["name"]
            base = m["base_url"].rstrip("/")
            clients[nm] = OpenAI(base_url=f"{base}/v1", api_key="EMPTY")
            model_to_gpu[nm] = gid

    model_names = list(clients.keys())

    # ---------------------------------
    # Load prompts
    # ---------------------------------
    df = pd.read_parquet(args.prompt_path)
    df = df.sample(n=args.num_prompts, random_state=42)
    prompts = list(df["prompt"])
    print(f"[Eval] Loaded {len(prompts)} prompts.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer_lock = threading.Lock()
    pattern = RequestPattern(args.pattern, args.rate)

    # SLO
    A_SLO, B_SLO = 0.5, 0.002
    slo = lambda p, d: A_SLO + B_SLO * (p + d)

    # ---------------------------------
    # Worker
    # ---------------------------------
    def worker(pid, prompt):
        hw = model_metrics
        p_tokens = len(prompt.split())

        best_model = None
        best_score = None
        best_router_quality = 0
        best_router_cost = 0
        best_pred_ttft = 0
        best_pred_tpot = 0
        best_gpu = None
        best_d_tokens = 0

        # Loop all models
        for m in model_names:
            gpu = model_to_gpu[m]
            snap = hw.get(m, {})

            # router scores
            r_quality, r_cost = router.compute(m, prompt)

            # hw features
            feat = {
                "p_tokens": p_tokens,
                "running_req_count": snap.get("num_requests_running", 0),
                "waiting_req_count": snap.get("num_requests_waiting", 0),
                "kv_cache_usage_perc": snap.get("kv_cache_usage_perc", 0.0),
                "model_id": m,
                "gpu_id": gpu,
            }

            # dynamic cost
            if args.use_hw_cost:
                pred_ttft, pred_tpot = cost_predictor(m, feat)
                dyn_cost = pred_ttft + pred_tpot
                final_cost = dyn_cost
            else:
                pred_ttft = pred_tpot = 0
                final_cost = r_cost

            score = r_quality - args.cost_lambda * final_cost

            if best_score is None or score > best_score:
                best_score = score
                best_model = m
                best_router_quality = r_quality
                best_router_cost = r_cost
                best_pred_ttft = pred_ttft
                best_pred_tpot = pred_tpot
                best_gpu = gpu

        # Execute best model
        client = clients[best_model]
        obs_ttft, obs_tpot, lat, d_tokens = send_request(client, best_model, prompt)
        lim = slo(p_tokens, d_tokens)
        slo_met = int(lat <= lim)

        row = {
            "run_id": RUN_ID,
            "router_id": args.router,
            "pattern_type": args.pattern,
            "arrival_rate": args.rate,
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt_id": pid,
            "model_id": best_model,
            "gpu_id": best_gpu,
            "p_tokens": p_tokens,
            "d_tokens": d_tokens,
            "running_req_count": hw.get(best_model, {}).get("num_requests_running", 0),
            "waiting_req_count": hw.get(best_model, {}).get("num_requests_waiting", 0),
            "kv_cache_usage_perc": hw.get(best_model, {}).get("kv_cache_usage_perc", 0.0),
            "router_quality": best_router_quality,
            "router_cost": best_router_cost,
            "pred_ttft": best_pred_ttft,
            "pred_tpot": best_pred_tpot,
            "final_cost": best_pred_ttft + best_pred_tpot if args.use_hw_cost else best_router_cost,
            "final_score": best_score,
            "observed_ttft": obs_ttft,
            "observed_tpot": obs_tpot,
            "latency_s": lat,
            "slo_limit_s": lim,
            "slo_met": slo_met
        }

        with writer_lock:
            file_exists = os.path.exists(args.output)
            with open(args.output, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if not file_exists:
                    w.writeheader()
                w.writerow(row)

    # ---------------------------------
    # Thread pool
    # ---------------------------------
    threads = []
    for i, prompt in enumerate(prompts):
        while len([t for t in threads if t.is_alive()]) >= 10:
            time.sleep(0.01)

        t = threading.Thread(target=worker, args=(i, prompt))
        threads.append(t)
        t.start()
        time.sleep(pattern.next_delay())

    for t in threads:
        t.join()

    print(f"[Eval] Done → saved to {args.output}")


if __name__ == "__main__":
    main()
