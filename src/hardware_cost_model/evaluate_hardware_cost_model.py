"""
evaluate_hardware_cost_model.py
Evaluate trained Hardware Cost Model on held-out data.
"""

import torch
import pandas as pd
import numpy as np
from joblib import load
from .model_utils import HardwareCostNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Load dataset + preprocessor
# -----------------------------
CSV_PATH = "data/hw_dataset_qween.csv"
df = pd.read_csv(CSV_PATH)

df["model_gpu"] = df["model_id"].astype(str) + "_" + df["gpu_id"].astype(str)
df["ttft_s"] = df["ttft_s"].clip(lower=1e-4)
df["tpot_s_per_token"] = df["tpot_s_per_token"].clip(lower=1e-4)
df["ttft_s_log"] = np.log(df["ttft_s"])
df["tpot_s_log"] = np.log(df["tpot_s_per_token"])

X_cols = ["p_tokens", "running_req_count", "waiting_req_count",
          "kv_cache_usage_perc", "ttft_avg", "itl_avg", "model_gpu"]

preproc = load("checkpoints/hardware_cost_model/preproc.joblib")
X = preproc.transform(df)
y = df[["ttft_s_log", "tpot_s_log"]].values

# Split same as training (seed=42)
from sklearn.model_selection import train_test_split
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 2. Load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HardwareCostNet(X.shape[1]).to(device)
model.load_state_dict(torch.load("checkpoints/hardware_cost_model/model.pt", map_location=device))
model.eval()

# -----------------------------
# 3. Inference + metrics
# -----------------------------
with torch.no_grad():
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    ttft_pred, tpot_pred = model(Xv)
    y_pred = np.column_stack([ttft_pred.cpu().numpy().flatten(),
                              tpot_pred.cpu().numpy().flatten()])

mae_ttft = mean_absolute_error(y_val[:, 0], y_pred[:, 0])
mae_tpot = mean_absolute_error(y_val[:, 1], y_pred[:, 1])
r2_ttft = r2_score(y_val[:, 0], y_pred[:, 0])
r2_tpot = r2_score(y_val[:, 1], y_pred[:, 1])

print("\n📊 Evaluation Results (log-scale):")
print(f"TTFT  → MAE={mae_ttft:.4f}, R²={r2_ttft:.4f}")
print(f"TPOT  → MAE={mae_tpot:.4f}, R²={r2_tpot:.4f}")

# Optionally back-transform to linear space
ttft_true = np.exp(y_val[:, 0])
ttft_pred_lin = np.exp(y_pred[:, 0])
tpot_true = np.exp(y_val[:, 1])
tpot_pred_lin = np.exp(y_pred[:, 1])

mae_ttft_lin = mean_absolute_error(ttft_true, ttft_pred_lin)
mae_tpot_lin = mean_absolute_error(tpot_true, tpot_pred_lin)

print("\n📈 Evaluation Results (original scale):")
print(f"TTFT  → MAE={mae_ttft_lin:.4f}s")
print(f"TPOT  → MAE={mae_tpot_lin:.6f}s/token")
