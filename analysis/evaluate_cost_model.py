"""
evaluate_hardware_cost_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hw_router.cost_predictor import HardwareCostPredictor
from hw_router.model_registry import get_model_id

# -----------------------------
# 0. Config
# -----------------------------
OUT_DIR = os.path.expanduser("~/data/cost_predictor/")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = "data/h100_full_sweep.csv"
MODEL_PATH = "checkpoints/hardware_cost_model/model.pt"
PREPROC_PATH = "checkpoints/hardware_cost_model/preproc.joblib"

sns.set(style="whitegrid", font_scale=1.4)
plt.rcParams["figure.figsize"] = (6, 5)


# -----------------------------
# 1. Load + normalize dataset
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Drop unused columns
df = df.drop(
    columns=["request_id", "timestamp", "prompt_id", "latency_s", "e2e_avg"],
    errors="ignore",
)

# 🔥 Enforce integer model_id everywhere
df["model_id"] = df["model_id"].map(get_model_id)

# Targets (log-scale)
df["ttft_s"] = df["ttft_s"].clip(lower=1e-4)
df["tpot_s_per_token"] = df["tpot_s_per_token"].clip(lower=1e-4)

df["ttft_s_log"] = np.log(df["ttft_s"])
df["tpot_s_log"] = np.log(df["tpot_s_per_token"])

# Split
indices = np.arange(len(df))
_, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
df_val = df.iloc[val_idx].reset_index(drop=True)

print(f"Loaded dataset: total={len(df)}, val={len(df_val)}")


# -----------------------------
# 2. Load predictor
# -----------------------------
predictor = HardwareCostPredictor(MODEL_PATH, PREPROC_PATH)


# -----------------------------
# 3. Evaluate predictor
# -----------------------------
ttft_true_lin = []
ttft_pred_lin = []
tpot_true_lin = []
tpot_pred_lin = []

ttft_true_log = []
ttft_pred_log = []
tpot_true_log = []
tpot_pred_log = []

p_tokens_list = []
kv_usage_list = []

for _, row in df_val.iterrows():

    model_id = int(row["model_id"])
    gpu_id = str(row["gpu_id"])   # kept as string only because predictor builds model_gpu="<id>_<gpu>"

    # True values
    ttft_true_lin.append(row["ttft_s"])
    tpot_true_lin.append(row["tpot_s_per_token"])
    ttft_true_log.append(row["ttft_s_log"])
    tpot_true_log.append(row["tpot_s_log"])

    # Build feature dict EXACTLY like training / eval_pipeline
    feat = {
        "p_tokens": int(row["p_tokens"]),
        "running_req_count": int(row["running_req_count"]),
        "waiting_req_count": int(row["waiting_req_count"]),
        "kv_cache_usage_perc": float(row["kv_cache_usage_perc"]),
        "ttft_avg": float(row.get("ttft_avg", 0.0)),
        "itl_avg": float(row.get("itl_avg", 0.0)),
        "model_id": model_id,   # integer, consistent everywhere
        "gpu_id": gpu_id,
    }

    # Prediction (🔥 now always model_id_int)
    ttft_hat, tpot_hat = predictor(model_id, feat)

    ttft_pred_lin.append(ttft_hat)
    tpot_pred_lin.append(tpot_hat)

    ttft_pred_log.append(np.log(max(ttft_hat, 1e-8)))
    tpot_pred_log.append(np.log(max(tpot_hat, 1e-8)))

    p_tokens_list.append(row["p_tokens"])
    kv_usage_list.append(row["kv_cache_usage_perc"])


# -----------------------------
# 4. Compute metrics
# -----------------------------
def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

metrics_df = pd.DataFrame([
    ["TTFT (log)", *metrics(ttft_true_log, ttft_pred_log).values()],
    ["TPOT (log)", *metrics(tpot_true_log, tpot_pred_log).values()],
    ["TTFT (orig)", *metrics(ttft_true_lin, ttft_pred_lin).values()],
    ["TPOT (orig)", *metrics(tpot_true_lin, tpot_pred_lin).values()],
], columns=["Metric", "MAE", "RMSE", "R2"])

metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_table.csv"), index=False)
print(metrics_df)


# -----------------------------
# 5. Plots
# -----------------------------
def scatter_plot(y_true, y_pred, title, fname):
    plt.figure()
    sns.scatterplot(x=y_true, y=y_pred, s=10, alpha=0.4)
    lim = [0, max(max(y_true), max(y_pred))]
    plt.plot(lim, lim, "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=300)
    plt.close()


def error_dist(y_true, y_pred, title, fname):
    plt.figure()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sns.histplot(y_pred - y_true, bins=50, kde=True)
    plt.title(title)
    plt.xlabel("Prediction Error")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=300)
    plt.close()



# Scatter plots
scatter_plot(ttft_true_lin, ttft_pred_lin, "TTFT Prediction", "ttft_pred.png")
scatter_plot(tpot_true_lin, tpot_pred_lin, "TPOT Prediction", "tpot_pred.png")

# Error distributions
error_dist(ttft_true_lin, ttft_pred_lin, "TTFT Error Distribution", "ttft_err.png")
error_dist(tpot_true_lin, tpot_pred_lin, "TPOT Error Distribution", "tpot_err.png")

print(f"\nAll outputs saved to {OUT_DIR}")
