import pandas as pd
import numpy as np

# =========================================================
#   Model Price Mapping (HuggingFace names)
# =========================================================

MODEL_PRICES = {
    "Qwen2.5-14B-Instruct": 0.22 / 1_000_000,
    "Phi-3-mini-128k-instruct": 0.10 / 1_000_000,
    "Llama-3.1-8B-Instruct": 0.03 / 1_000_000,
    "Qwen2.5-3B-Instruct": 0.05 / 1_000_000,
    "Mistral-7B-Instruct-v0.3": 0.20 / 1_000_000,
}

# =========================================================
#   Local Model → HF Mapping (from your model_maps.py)
# =========================================================

LOCAL_MODEL_TO_HUGGINGFACE_NAME = {
    "/home/ah872032/models/qwen14b" : "Qwen2.5-14B-Instruct",
    "/home/ah872032/models/phi3-mini" : "Phi-3-mini-128k-instruct",
    "/home/ah872032/models/llama3-8b" : "Llama-3.1-8B-Instruct",
    "/home/ah872032/models/qwen3b" : "Qwen2.5-3B-Instruct",
    "/home/ah872032/models/mistral7b" : "Mistral-7B-Instruct-v0.3",
}

DATA_PATH = "data/h100_full_sweep.csv"

# =========================================================
#   Load Dataset
# =========================================================

print(f"Loading training data: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

required_cols = ["latency_s", "d_tokens", "model_id"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}'")

# =========================================================
#   Compute latency percentile constant (log-scaling)
# =========================================================

df["latency_s"] = df["latency_s"].astype(float).clip(lower=1e-6)
df["log_latency"] = np.log1p(df["latency_s"])

latency_p95_log = float(np.percentile(df["log_latency"], 95))

print(f"\nlatency_p95_log = {latency_p95_log:.6f}")

# =========================================================
#   Compute static CARROT cost percentile constant
# =========================================================

def compute_static_cost(row):
    model_local = row["model_id"]
    d = float(row["d_tokens"])

    if model_local not in LOCAL_MODEL_TO_HUGGINGFACE_NAME:
        raise ValueError(f"Unknown model path: {model_local}")

    hf = LOCAL_MODEL_TO_HUGGINGFACE_NAME[model_local]

    if hf not in MODEL_PRICES:
        raise ValueError(f"Missing price for model: {hf}")

    return d * MODEL_PRICES[hf]

df["static_cost"] = df.apply(compute_static_cost, axis=1)

static_cost_p95 = float(np.percentile(df["static_cost"], 95))

print(f"static_cost_p95 = {static_cost_p95:.12f}")

# =========================================================
#   Summary
# =========================================================

print("\n=== FINAL NORMALIZATION CONSTANTS ===")
print(f"latency_p95_log = {latency_p95_log:.6f}")
print(f"static_cost_p95 = {static_cost_p95:.12f}")
print("=====================================\n")
