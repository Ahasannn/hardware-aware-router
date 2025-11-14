# eval_pipeline.py

import os, csv, uuid, time, datetime, yaml, threading
import pandas as pd
from openai import OpenAI

from .routers import (
    BaselineRouter, RandomRouter,
    RoundRobinRouter, CarrotRouter
)
from .cost_model import HardwareCostPredictor
from .metrics_watcher import model_metrics, start_metrics_watcher
from .load_patterns import RequestPattern
from baselines.carrot import load_carrot_router

from .model_maps import (
    get_model_id,
    get_model_hugging_face_name
)

# -------------------------
# CSV Schema
# -------------------------
CSV_FIELDS = [
    "run_id", "router_id",
    "pattern_type", "arrival_rate",
    "request_id", "timestamp", "prompt_id",

    # -------- Router-only (no HW cost) --------
    "router_only_model",
    "router_only_gpu_id",
    "router_only_quality",
    "router_only_cost",
    "router_only_score",
    "router_only_hw_running",
    "router_only_hw_waiting",
    "router_only_hw_kv_cache",

    # -------- HW-aware selected model --------
    "selected_model",
    "selected_gpu_id",
    "selected_quality",
    "selected_cost",       # static router cost
    "selected_hw_cost",    # our predicted cost (latency-based), may be empty if HW off
    "selected_score",
    "selected_hw_running",
    "selected_hw_waiting",
    "selected_hw_kv_cache",

    # -------- Cost model predictions --------
    "pred_ttft",
    "pred_tpot",
    "pred_total_latency",
    "predicted_length",

    # -------- Observed latency --------
    "p_tokens",
    "d_tokens",
    "observed_ttft",
    "observed_tpot",
    "latency_s",

    # -------- SLO --------
    "slo_limit_s",
    "slo_met",
    "slo_margin_s",

    # -------- Router overhead --------
    "router_compute_ms",
]

# -------------------------
# Send Request
# -------------------------
def send_request(client, model, prompt):
    start = time.time()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=1024,
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
    tpot = (lat - ttft) / max(ntoks, 1) if ttft > 0 else 0.0
    return ttft, tpot, lat, ntoks

# -------------------------
# GPU Utilization Monitor
# -------------------------
def start_gpu_monitor(model_to_gpu, gpu_stats_list, interval=1.0):
    def monitor():
        gpu_ids = sorted(set(model_to_gpu.values()))
        while True:
            now = time.time()
            for gid in gpu_ids:
                running = 0
                waiting = 0
                kv_cache = 0.0

                for model, mg in model_to_gpu.items():
                    if mg != gid:
                        continue
                    snap = model_metrics.get(model, {})
                    running += snap.get("num_requests_running", 0)
                    waiting += snap.get("num_requests_waiting", 0)
                    kv_cache = max(kv_cache, snap.get("kv_cache_usage_perc", 0.0))

                gpu_stats_list.append({
                    "timestamp": now,
                    "gpu_id": gid,
                    "running_req": running,
                    "waiting_req": waiting,
                    "kv_cache_usage": kv_cache,
                })

            time.sleep(interval)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()

# -------------------------
# Main
# -------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt_path", default="data/prompts/mixed_prompts_eval.parquet")
    parser.add_argument("--output", default="data/eval_results/eval_results.csv")
    parser.add_argument("--router", choices=["baseline", "random", "rr", "carrot"], required=True)
    parser.add_argument("--use_hw_cost", action="store_true")
    parser.add_argument("--cost_lambda", type=float, default=1.0)
    parser.add_argument("--num_prompts", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--pattern", default="poisson",
                        choices=["poisson", "microburst", "sustained"])
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--model_path", default="checkpoints/hardware_cost_model/model.pt")
    parser.add_argument("--preproc_path", default="checkpoints/hardware_cost_model/preproc.joblib")
    args = parser.parse_args()

    RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[Eval] Router={args.router}, HW-aware={args.use_hw_cost}, run_id={RUN_ID}")

    # --------------------------
    # Routers
    # --------------------------
    carrot_router = CarrotRouter(load_carrot_router("checkpoints/carrot", model_type="linear"))

    if args.router == "baseline":
        router = BaselineRouter()
    elif args.router == "random":
        router = RandomRouter()
    elif args.router == "rr":
        router = RoundRobinRouter()
    elif args.router == "carrot":
        router = carrot_router
    else:
        raise ValueError("Unknown router")

    # --------------------------
    # Cost Model
    # --------------------------
    cost_predictor = HardwareCostPredictor(args.model_path, args.preproc_path) \
                     if args.use_hw_cost else None

    # --------------------------
    # Configuration (vLLM)
    # --------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    clients, model_to_gpu, model_url_map = {}, {}, {}
    for g, models in cfg["gpus"].items():
        g_id = int(g)
        for m in models:
            name = m["name"]
            base = m["base_url"].rstrip("/")
            clients[name] = OpenAI(base_url=f"{base}/v1", api_key="EMPTY")
            model_to_gpu[name] = g_id
            model_url_map[name] = f"{base}/metrics"

    model_names = list(clients.keys())

    # Start metrics watcher & GPU monitor
    start_metrics_watcher(model_url_map, interval=args.interval)

    gpu_stats = []
    start_gpu_monitor(model_to_gpu, gpu_stats, interval=1.0)

    # Prompts
    df = pd.read_parquet(args.prompt_path).sample(n=args.num_prompts, random_state=42)
    prompts = list(df["prompt"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer_lock = threading.Lock()

    pattern = RequestPattern(args.pattern, args.rate)

    # SLO: linear function of total tokens
    slo = lambda p, d: 0.5 + 0.002 * (p + d)

    # ------------------------
    # Worker (per prompt)
    # ------------------------
    def worker(pid, prompt):
        p_tokens = len(prompt.split())
        hw = model_metrics

        t_router_start = time.time()

        # Router-only (no HW cost) best choice
        best_router_only_score = None
        router_only_choice = None

        # HW-aware best choice
        best_hw_score = None
        hw_choice = None

        for m in model_names:
            gpu = model_to_gpu[m]
            snap = hw.get(m, {})
            m_hugging_face_name = get_model_hugging_face_name(m)

            # router's intrinsic quality & static cost
            r_quality, r_cost = router.compute(m_hugging_face_name, prompt)

            # Router-only score (no HW cost)
            router_only_score = r_quality - args.cost_lambda * r_cost

            # record router-only best
            if (best_router_only_score is None) or (router_only_score > best_router_only_score):
                best_router_only_score = router_only_score
                router_only_choice = {
                    "model": m,
                    "gpu": gpu,
                    "quality": r_quality,
                    "cost": r_cost,
                    "score": router_only_score,
                    "hw_running": snap.get("num_requests_running", 0),
                    "hw_waiting": snap.get("num_requests_waiting", 0),
                    "hw_kv": snap.get("kv_cache_usage_perc", 0.0),
                }

            # HW-aware branch
            pred_ttft = 0.0
            pred_tpot = 0.0
            pred_total = 0.0
            predicted_length = 0

            if cost_predictor:
                # Features for cost model
                model_id_int = get_model_id(m)
                feat = {
                    "p_tokens": p_tokens,
                    "running_req_count": snap.get("num_requests_running", 0),
                    "waiting_req_count": snap.get("num_requests_waiting", 0),
                    "kv_cache_usage_perc": snap.get("kv_cache_usage_perc", 0.0),
                    "ttft_avg": snap.get("ttft_avg", 0.0),
                    "itl_avg": snap.get("itl_avg", 0.0),
                    "model_id": model_id_int,
                    "gpu_id": str(gpu),
                }

                # predict TTFT & TPOT
                pred_ttft, pred_tpot = cost_predictor(model_id_int, feat)

                # predict length via carrot (fallback to 100 if anything fails)
                try:
                    predicted_length = int(carrot_router.length_predictor(m_hugging_face_name, prompt))
                except Exception:
                    predicted_length = 100

                pred_total = pred_ttft + predicted_length * pred_tpot
                hw_cost = pred_total
                hw_score = r_quality - args.cost_lambda * hw_cost
            else:
                # if HW cost disabled, HW-aware == router-only
                hw_cost = r_cost
                hw_score = router_only_score

            # record HW-aware best
            if (best_hw_score is None) or (hw_score > best_hw_score):
                best_hw_score = hw_score
                hw_choice = {
                    "model": m,
                    "gpu": gpu,
                    "quality": r_quality,
                    "cost": r_cost,         # static
                    "hw_cost": hw_cost,     # predicted or static
                    "score": hw_score,
                    "hw_running": snap.get("num_requests_running", 0),
                    "hw_waiting": snap.get("num_requests_waiting", 0),
                    "hw_kv": snap.get("kv_cache_usage_perc", 0.0),
                    "pred_ttft": pred_ttft,
                    "pred_tpot": pred_tpot,
                    "pred_total": pred_total,
                    "predicted_length": predicted_length,
                }

        router_time_ms = (time.time() - t_router_start) * 1000.0

        # fallbacks (should never happen, but be safe)
        if router_only_choice is None or hw_choice is None:
            return

        # Execute HW-aware selected model
        selected_model = hw_choice["model"]
        client = clients[selected_model]
        obs_ttft, obs_tpot, lat, d_tokens = send_request(client, selected_model, prompt)

        lim = slo(p_tokens, d_tokens)
        slo_met = int(lat <= lim)
        slo_margin = lim - lat

        row = {
            "run_id": RUN_ID,
            "router_id": args.router,
            "pattern_type": args.pattern,
            "arrival_rate": args.rate,
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt_id": pid,

            # Router-only choice
            "router_only_model": get_model_hugging_face_name(router_only_choice["model"]),
            "router_only_gpu_id": router_only_choice["gpu"],
            "router_only_quality": router_only_choice["quality"],
            "router_only_cost": router_only_choice["cost"],
            "router_only_score": router_only_choice["score"],
            "router_only_hw_running": router_only_choice["hw_running"],
            "router_only_hw_waiting": router_only_choice["hw_waiting"],
            "router_only_hw_kv_cache": router_only_choice["hw_kv"],

            # HW-aware (selected) choice
            "selected_model": get_model_hugging_face_name(hw_choice["model"]),
            "selected_gpu_id": hw_choice["gpu"],
            "selected_quality": hw_choice["quality"],
            "selected_cost": hw_choice["cost"],
            "selected_hw_cost": hw_choice["hw_cost"],
            "selected_score": hw_choice["score"],
            "selected_hw_running": hw_choice["hw_running"],
            "selected_hw_waiting": hw_choice["hw_waiting"],
            "selected_hw_kv_cache": hw_choice["hw_kv"],

            # Cost model predictions
            "pred_ttft": hw_choice["pred_ttft"],
            "pred_tpot": hw_choice["pred_tpot"],
            "pred_total_latency": hw_choice["pred_total"],
            "predicted_length": hw_choice["predicted_length"],

            # Observed latency
            "p_tokens": p_tokens,
            "d_tokens": d_tokens,
            "observed_ttft": obs_ttft,
            "observed_tpot": obs_tpot,
            "latency_s": lat,

            # SLO
            "slo_limit_s": lim,
            "slo_met": slo_met,
            "slo_margin_s": slo_margin,

            # Router overhead
            "router_compute_ms": router_time_ms,
        }

        with writer_lock:
            file_exists = os.path.exists(args.output)
            with open(args.output, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if not file_exists:
                    w.writeheader()
                w.writerow(row)

    # ------------------------
    # Thread pool
    # ------------------------
    threads = []
    for i, prompt in enumerate(prompts):
        while len([t for t in threads if t.is_alive()]) >= args.concurrency:
            time.sleep(0.05)

        t = threading.Thread(target=worker, args=(i, prompt))
        threads.append(t)
        t.start()
        time.sleep(pattern.next_delay())

    for t in threads:
        t.join()

    # Save GPU stats
    gpu_out = args.output.replace(".csv", "_gpu_util.csv")
    pd.DataFrame(gpu_stats).to_csv(gpu_out, index=False)

    print(f"[Eval] Done → saved to {args.output}")
    print(f"[Eval] GPU utilization stats → {gpu_out}")


if __name__ == "__main__":
    main()
