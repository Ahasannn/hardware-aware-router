import os, time, json, threading, queue, uuid, datetime, yaml
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from openai import OpenAI

from src.hardware_cost_model.routers import CarrotRouter
from baselines.carrot import load_carrot_router

from src.hardware_cost_model.metrics_watcher import model_metrics, start_metrics_watcher
from src.hardware_cost_model.load_patterns import RequestPattern

from src.hardware_cost_model.model_maps import get_model_id, get_model_hugging_face_name


# -------------------------
# Send request (same logic)
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
    tpot = (lat - ttft) / max(ntoks, 1) if ttft > 0 else 0
    return ttft, tpot, lat, ntoks


# -------------------------
# Motivation Sweep
# -------------------------
def run_motivation_sweep(
        config,
        prompt_path,
        arrival_rates,
        concurrency,
        interval,
        output_csv,
        num_prompts):            # NEW
    # -------------------------
    # Load config
    # -------------------------
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    clients, model_to_gpu = {}, {}
    model_names = []
    model_url_map = {}

    for g, models in cfg["gpus"].items():
        g_id = int(g)
        for m in models:
            name = m["name"]
            base = m["base_url"].rstrip("/")

            clients[name] = OpenAI(base_url=f"{base}/v1", api_key="EMPTY")
            model_to_gpu[name] = g_id
            model_names.append(name)
            model_url_map[name] = f"{base}/metrics"

    # Start metrics watcher
    start_metrics_watcher(model_url_map, interval=interval)

    # -------------------------
    # Router
    # -------------------------
    carrot_router = CarrotRouter(load_carrot_router("checkpoints/carrot", model_type="linear"))

    # -------------------------
    # Prompts (shuffle + limit)
    # -------------------------
    df = pd.read_parquet(prompt_path)

    # MUST contain: prompt, embedding (list or np array)
    if "carrot_emb" not in df.columns:
        raise ValueError("Your parquet file must contain `embedding` column.")

    # Shuffle & limit
    df = df.sample(frac=1.0, random_state=42).head(num_prompts)

    # Store lists
    prompts = df["prompt"].tolist()
    embeddings = df["carrot_emb"].tolist()       # <-- NEW
    N = len(prompts)


    final_results = []

    # ============================================================
    # Sweep through arrival rates
    # ============================================================
    for rate in arrival_rates:

        print(f"\n========== Running λ = {rate} ==========")

        #pattern = RequestPattern("poisson", rate)
        pattern = RequestPattern("microburst", rate)
        

        # Stats
        model_dist = Counter()
        gpu_dist = Counter()

        total_waited = 0
        model_wait_lists = defaultdict(list)     # NEW

        kv_obs = defaultdict(list)
        latencies = []

        q = queue.Queue()

        # -------------------------
        # Worker
        # -------------------------
        def worker():
            nonlocal total_waited

            while True:
                item = q.get()
                if item is None:
                    return

                pid, prompt = item

                # choose model via Carrot
                best_score = -1e9
                chosen = None

                for m in model_names:
                    hf_name = get_model_hugging_face_name(m)
                    emb = np.array(embeddings[pid])
                    quality, cost = carrot_router.compute_from_embedding(hf_name, emb)
                    score = quality - cost
                    if score > best_score:
                        best_score = score
                        chosen = m

                model_name = chosen
                gpu_id = model_to_gpu[model_name]

                # metrics BEFORE dispatch
                snap = model_metrics.get(model_name, {})
                waiting = snap.get("num_requests_waiting", 0)
                kv = snap.get("kv_cache_usage_perc", 0.0)

                if waiting > 0:
                    total_waited += waiting
                model_wait_lists[model_name].append(waiting)   # NEW

                kv_obs[model_name].append(kv)

                # issue request
                ttft, tpot, lat, d = send_request(clients[model_name], model_name, prompt)
                latencies.append(lat)

                model_dist[model_name] += 1
                gpu_dist[gpu_id] += 1

                q.task_done()

        # -------------------------
        # Thread pool
        # -------------------------
        threads = []
        for _ in range(concurrency):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        # feed requests with Poisson delay
        for i, prompt in enumerate(prompts):
            q.put((i, prompt))
            time.sleep(pattern.next_delay())

        # stop workers
        for _ in threads:
            q.put(None)
        for t in threads:
            t.join()

        # aggregate stats
        avg_kv = {m: float(np.mean(v)) if v else 0.0 for m, v in kv_obs.items()}
        avg_wait = {m: float(np.mean(v)) if v else 0.0 for m, v in model_wait_lists.items()}   # NEW

        # Convert model names to HF names for nicer output
        hf_model_dist = {}
        hf_avg_wait = {}
        hf_avg_kv = {}

        for m in model_names:
            hf = get_model_hugging_face_name(m)   # <-- NEW

            # model_distribution: fill missing with 0
            hf_model_dist[hf] = model_dist.get(m, 0)

            # waiting: fill missing with 0
            waits = model_wait_lists.get(m, [])
            hf_avg_wait[hf] = float(np.mean(waits)) if waits else 0.0

            # kv-cache: fill missing with 0
            kvs = kv_obs.get(m, [])
            hf_avg_kv[hf] = float(np.mean(kvs)) if kvs else 0.0


        result = {
            "arrival_rate": rate,
            "num_requests": N,

            # HF names instead of ugly local paths
            "model_distribution": hf_model_dist,
            "gpu_distribution": dict(gpu_dist),

            # Per-model queue imbalance (motivation)
            "avg_waiting_per_model": hf_avg_wait,

            # Per-model KV cache pressure
            "avg_kv_cache": hf_avg_kv,

            # Latency
            "avg_latency": float(np.mean(latencies)) if latencies else 0.0,
            "p95_latency": float(np.percentile(latencies, 95)) if latencies else 0.0
        }

        print(json.dumps(result, indent=2))
        final_results.append(result)

        # -----------------------------------------
        # Save after every arrival_rate (append)
        # -----------------------------------------
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df_tmp = pd.DataFrame([result])

        # If file does not exist → write header
        if not os.path.isfile(output_csv):
            df_tmp.to_csv(output_csv, index=False)
        else:
            df_tmp.to_csv(output_csv, mode='a', header=False, index=False)

        print(f"✔ Saved partial result for λ={rate} → {output_csv}")


    print(f"\nSaved motivation sweep → {output_csv}")


# -------------------------
# Main Entry
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gpu_model_map_h100.yaml")
    parser.add_argument("--prompt_path", default="data/prompts/mixed_prompts_eval_with_prompt_embeddings.parquet")
    parser.add_argument("--output", default="data/motivation_sweep.csv")
    parser.add_argument("--concurrency", type=int, default=120)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--arrival_rates", nargs="+", type=float, default=[3,9,6,12,15,18])
    parser.add_argument("--num_prompts", type=int, default=300)     
    args = parser.parse_args()

    run_motivation_sweep(
        config=args.config,
        prompt_path=args.prompt_path,
        arrival_rates=args.arrival_rates,
        concurrency=args.concurrency,
        interval=args.interval,
        output_csv=args.output,
        num_prompts=args.num_prompts         
    )
