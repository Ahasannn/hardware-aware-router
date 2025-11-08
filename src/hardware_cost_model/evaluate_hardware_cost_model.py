"""
evaluate_hardware_cost_model.py
Evaluate trained Hardware Cost Model on a new CSV file.
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model_utils import HardwareCostNet, load_cost_model


# -----------------------------
# 1. Load trained model
# -----------------------------
preproc, model, device = load_cost_model("checkpoints/hardware_cost_model")

# -----------------------------
# 2. Load evaluation dataset
# -----------------------------
EVAL_CSV = "data/hw_dataset.csv"   # change path if needed
df = pd.read_csv(EVAL_CSV)

df["model_gpu"] = df["model_id"].astype(str) + "_" + df["gpu_id"].astype(str)
X = df[["p_tokens", "running_req_count", "waiting_req_count",
        "kv_cache_usage_perc", "ttft_avg", "itl_avg", "model_gpu"]]
y_true_ttft = df["ttft_s"].values
y_true_tpot = df["tpot_s_per_token"].values

# -----------------------------
# 3. Predict all rows
# -----------------------------
X_proc = preproc.transform(X)
X_t = torch.tensor(X_proc, dtype=torch.float32).to(device)
with torch.no_grad():
    ttft_hat, tpot_hat = model(X_t)

ttft_pred = torch.exp(ttft_hat).cpu().numpy().flatten()
tpot_pred = torch.exp(tpot_hat).cpu().numpy().flatten()

# -----------------------------
# 4. Compute metrics
# -----------------------------
def report_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Metrics:")
    print(f"  MAE: {mae:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  R² : {r2:.4f}")

report_metrics(y_true_ttft, ttft_pred, "TTFT")
report_metrics(y_true_tpot, tpot_pred, "TPOT")

corr_ttft = np.corrcoef(y_true_ttft, ttft_pred)[0, 1]
corr_tpot = np.corrcoef(y_true_tpot, tpot_pred)[0, 1]
print(f"\nCorrelation (TTFT): {corr_ttft:.4f}")
print(f"Correlation (TPOT): {corr_tpot:.4f}")
print("\n✅ Evaluation complete.")
