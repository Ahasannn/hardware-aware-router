import time
import random
from tqdm import tqdm

from baselines.carrot import load_carrot_router
from hw_router.routers import CarrotRouter
from hw_router.cost_predictor import HardwareCostPredictor
from hw_router.model_registry import get_model_id, get_model_hugging_face_name


# ---------------------------------------------------------
# Generate random prompt based on token count
# ---------------------------------------------------------
def generate_prompt(token_count: int):
    return " ".join([f"word{i}" for i in range(token_count)])


# ---------------------------------------------------------
# Dummy hardware features (no runtime metrics needed)
# ---------------------------------------------------------
def fake_hw_features(p, model_id):
    return {
        "p_tokens": p,
        "running_req_count": 0,
        "waiting_req_count": 0,
        "kv_cache_usage_perc": 0.20,
        "ttft_avg": 0.15,
        "itl_avg": 0.03,
        "model_id": model_id,
        "gpu_id": "0",
    }


# =========================================================
# Benchmark CARROT-only latency
# =========================================================
def benchmark_carrot_only(carrot_router, model_list, n=100):
    latencies = []

    for _ in tqdm(range(n), desc="CARROT Only"):
        p = 1000
        prompt = generate_prompt(p)
        model_name = random.choice(model_list)

        start = time.time()
        
        hf_model_name = get_model_hugging_face_name(model_name)
        q, c = carrot_router.compute(hf_model_name, prompt)
    
        end = time.time()
        latencies.append(end - start)

    return sum(latencies) / len(latencies)


# =========================================================
# Benchmark CARROT + HW CostModel latency
# =========================================================
def benchmark_carrot_plus_costmodel(carrot_router, hw_predictor, model_list, n=100):
    latencies = []

    for _ in tqdm(range(n), desc="CARROT + CostModel"):
        p = 1000
        prompt = generate_prompt(p)
        model_name = random.choice(model_list)
        model_id_int = get_model_id(model_name)
        hf_model_name = get_model_hugging_face_name(model_name)

        feat = fake_hw_features(p, model_id_int)

        start = time.time()

        # CARROT cost + quality + predicted length
        q, c = carrot_router.compute(hf_model_name, prompt)
        

        # HW latency predictor (extra overhead)
        pred_ttft, pred_tpot = hw_predictor(model_id_int, feat)

        end = time.time()
        latencies.append(end - start)

    return sum(latencies) / len(latencies)


# =========================================================
# Main
# =========================================================
def main():
    print("Loading CARROT + CostModel...")

    carrot_raw = load_carrot_router("checkpoints/carrot", model_type="linear")
    carrot_router = CarrotRouter(carrot_raw)

    hw_predictor = HardwareCostPredictor(
        "checkpoints/hardware_cost_model/model.pt",
        "checkpoints/hardware_cost_model/preproc.joblib"
    )

    model_list = [
        "llama3-8b",
        "mistral7b",
        "phi3-mini",
        "qwen3b",
        "qwen14b",
    ]


    print("\nBenchmarking...\n")

    carrot_lat = benchmark_carrot_only(carrot_router, model_list)
    hw_lat = benchmark_carrot_plus_costmodel(carrot_router, hw_predictor, model_list)

    print("\n============ RESULTS ============")
    print(f"CARROT only:          {carrot_lat * 1e3:.3f} ms")
    print(f"CARROT + CostModel:   {hw_lat * 1e3:.3f} ms")
    print(f"Extra overhead:        {(hw_lat - carrot_lat) * 1e3:.3f} ms")
    print("=================================\n")


if __name__ == "__main__":
    main()
