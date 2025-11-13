"""
evaluate_hardware_cost_model.py
Evaluate trained Hardware Cost Model on held-out data and generate paper-ready figures/tables.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .cost_model import HardwareCostPredictor

# -----------------------------
# 0. Setup
# -----------------------------
OUT_DIR = os.path.expanduser("~/data/cost_predictor/")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = "data/hw_dataset_qwen_sweep_with_output.csv"
MODEL_PATH = "checkpoints/hardware_cost_model/model.pt"
PREPROC_PATH = "checkpoints/hardware_cost_model/preproc.joblib"

sns.set(style="whitegrid", font_scale=1.4)
plt.rcParams["figure.figsize"] = (6, 5)


# -----------------------------
# 1. Load dataset + preprocess
# -----------------------------
df = pd.read_csv(CSV_PATH)

# SAME cleanup as training
df = df.drop(
    columns=["request_id", "timestamp", "prompt_id", "latency_s", "e2e_avg"],
    errors="ignore",
)

# Log-scale targets (same as training)
df["ttft_s"] = df["ttft_s"].clip(lower=1e-4)
df["tpot_s_per_token"] = df["tpot_s_per_token"].clip(lower=1e-4)
df["ttft_s_log"] = np.log(df["ttft_s"])
df["tpot_s_log"] = np.log(df["tpot_s_per_token"])

# Train/val split on indices (to reuse df directly)
indices = np.arange(len(df))
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=42
)

df_val = df.iloc[val_idx].reset_index(drop=True)

print(f"✅ Loaded dataset: total={len(df)}, val={len(df_val)}")


# -----------------------------
# 2. Load cost model
# -----------------------------
predictor = HardwareCostPredictor(MODEL_PATH, PREPROC_PATH)


# -----------------------------
# 3. Inference on validation set
# -----------------------------
ttft_true_log, tpot_true_log = [], []
ttft_pred_log, tpot_pred_log = []

ttft_true_lin, tpot_true_lin = [], []
ttft_pred_lin, tpot_pred_lin = []

p_tokens_list = []
kv_usage_list = []

for _, row in df_val.iterrows():
    model_name = row["model_id"]
    gpu_id = str(row["gpu_id"])

    # Ground truth (log + original scale)
    ttft_true_log.append(row["ttft_s_log"])
    tpot_true_log.append(row["tpot_s_log"])
    ttft_true_lin.append(row["ttft_s"])
    tpot_true_lin.append(row["tpot_s_per_token"])

    # Features for predictor (must match training schema)
    feat = {
        "p_tokens": int(row["p_tokens"]),
        "running_req_count": int(row["running_req_count"]),
        "waiting_req_count": int(row["waiting_req_count"]),
        "kv_cache_usage_perc": float(row["kv_cache_usage_perc"]),
        "ttft_avg": float(row.get("ttft_avg", 0.0)),
        "itl_avg": float(row.get("itl_avg", 0.0)),
        "model_id": model_name,
        "gpu_id": gpu_id,
    }

    ttft_hat, tpot_hat = predictor(model_name, feat)

    # Cache predictions (orig + log)
    ttft_pred_lin.append(ttft_hat)
    tpot_pred_lin.append(tpot_hat)

    ttft_pred_log.append(np.log(max(ttft_hat, 1e-8)))
    tpot_pred_log.append(np.log(max(tpot_hat, 1e-8)))

    # For feature-error analysis
    p_tokens_list.append(row["p_tokens"])
    kv_usage_list.append(row["kv_cache_usage_perc"])

ttft_true_log = np.array(ttft_true_log)
tpot_true_log = np.array(tpot_true_log)
ttft_pred_log = np.array(ttft_pred_log)
tpot_pred_log = np.array(tpot_pred_log)

ttft_true_lin = np.array(ttft_true_lin)
tpot_true_lin = np.array(tpot_true_lin)
ttft_pred_lin = np.array(ttft_pred_lin)
tpot_pred_lin = np.array(tpot_pred_lin)

p_tokens_arr = np.array(p_tokens_list)
kv_usage_arr = np.array(kv_usage_list)


# -----------------------------
# 4. Metrics
# -----------------------------
def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


log_ttft = metrics(ttft_true_log, ttft_pred_log)
log_tpot = metrics(tpot_true_log, tpot_pred_log)

lin_ttft = metrics(ttft_true_lin, ttft_pred_lin)
lin_tpot = metrics(tpot_true_lin, tpot_pred_lin)

metrics_df = pd.DataFrame(
    [
        ["TTFT (log)", *log_ttft.values()],
        ["TPOT (log)", *log_tpot.values()],
        ["TTFT (orig)", *lin_ttft.values()],
        ["TPOT (orig)", *lin_tpot.values()],
    ],
    columns=["Metric", "MAE", "RMSE", "R2"],
)
metrics_path = os.path.join(OUT_DIR, "metrics_table.csv")
metrics_df.to_csv(metrics_path, index=False)
print(metrics_df)


# -----------------------------
# 5. Visualization helpers
# -----------------------------
def scatter_plot(y_true, y_pred, title, fname):
    plt.figure()
    sns.scatterplot(x=y_true, y=y_pred, s=10, alpha=0.4)
    lim = [0, float(max(y_true.max(), y_pred.max()))]
    plt.plot(lim, lim, "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=300)
    plt.close()


def error_dist(y_true, y_pred, title, fname):
    plt.figure()
    sns.histplot(y_pred - y_true, bins=50, kde=True)
    plt.title(title)
    plt.xlabel("Prediction Error")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=300)
    plt.close()


def feature_corr_plot(feature, y_true, y_pred, fname, xlabel=None):
    plt.figure()
    plt.scatter(feature, np.abs(y_true - y_pred), s=8, alpha=0.3)
    plt.xlabel(xlabel if xlabel is not None else "feature")
    plt.ylabel("|Error|")
    plt.title(f"Error vs {xlabel if xlabel is not None else 'feature'}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=300)
    plt.close()


# -----------------------------
# 6. Plots
# -----------------------------
# Scatter & error plots (original scale)
scatter_plot(
    ttft_true_lin,
    ttft_pred_lin,
    "TTFT Prediction (Original Scale)",
    "ttft_true_vs_pred.png",
)
scatter_plot(
    tpot_true_lin,
    tpot_pred_lin,
    "TPOT Prediction (Original Scale)",
    "tpot_true_vs_pred.png",
)

error_dist(
    ttft_true_lin,
    ttft_pred_lin,
    "TTFT Error Distribution",
    "ttft_error_dist.png",
)
error_dist(
    tpot_true_lin,
    tpot_pred_lin,
    "TPOT Error Distribution",
    "tpot_error_dist.png",
)

# Feature sensitivity (using RAW features, which is fine)
feature_corr_plot(
    p_tokens_arr,
    ttft_true_lin,
    ttft_pred_lin,
    "ttft_error_vs_p_tokens.png",
    xlabel="p_tokens",
)

feature_corr_plot(
    kv_usage_arr,
    ttft_true_lin,
    ttft_pred_lin,
    "ttft_error_vs_kv_cache.png",
    xlabel="kv_cache_usage_perc",
)


# -----------------------------
# 7. Sample predictions
# -----------------------------
sample_df = pd.DataFrame(
    {
        "ttft_true_s": ttft_true_lin,
        "ttft_pred_s": ttft_pred_lin,
        "tpot_true_s_per_tok": tpot_true_lin,
        "tpot_pred_s_per_tok": tpot_pred_lin,
    }
)
sample_df["ttft_error_pct"] = 100 * (
    (sample_df["ttft_pred_s"] - sample_df["ttft_true_s"])
    / sample_df["ttft_true_s"].clip(lower=1e-6)
)
sample_df["tpot_error_pct"] = 100 * (
    (sample_df["tpot_pred_s_per_tok"] - sample_df["tpot_true_s_per_tok"])
    / sample_df["tpot_true_s_per_tok"].clip(lower=1e-6)
)

sample_df.head(50).to_csv(
    os.path.join(OUT_DIR, "sample_predictions.csv"), index=False
)

print(f"\n✅ All figures and tables saved in {OUT_DIR}")
