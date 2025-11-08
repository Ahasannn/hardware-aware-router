"""
train_hardware_cost_model.py
Train Hardware Cost Model to predict TTFT & TPOT.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump
from model_utils import HardwareCostNet


# -----------------------------
# 1. Load and preprocess dataset
# -----------------------------
CSV_PATH = "data/hw_dataset_highload.csv"
df = pd.read_csv(CSV_PATH)

# Drop irrelevant columns
df = df.drop(columns=[
    "request_id", "timestamp", "prompt_id", "latency_s", "e2e_avg"
], errors="ignore")

# Combine model_id + gpu_id
df["model_gpu"] = df["model_id"].astype(str) + "_" + df["gpu_id"].astype(str)
df = df.drop(columns=["gpu_id"], errors="ignore")

# Log-scale targets
df["ttft_s"] = df["ttft_s"].clip(lower=1e-4)
df["tpot_s_per_token"] = df["tpot_s_per_token"].clip(lower=1e-4)
df["ttft_s_log"] = np.log(df["ttft_s"])
df["tpot_s_log"] = np.log(df["tpot_s_per_token"])

# Feature sets
num_cols = ["p_tokens", "running_req_count", "waiting_req_count",
            "kv_cache_usage_perc", "ttft_avg", "itl_avg"]
cat_cols = ["model_gpu"]

preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

X = preproc.fit_transform(df)
y = df[["ttft_s_log", "tpot_s_log"]].values
dump(preproc, "checkpoints/hardware_cost_model/preproc.joblib")

print(f"✅ Preprocessing done | Features={X.shape[1]} | Samples={len(df)}")

# -----------------------------
# 2. Training setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=256, shuffle=True)

model = HardwareCostNet(X_tensor.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# -----------------------------
# 3. Training loop
# -----------------------------
for epoch in range(25):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        ttft_pred, tpot_pred = model(xb)
        loss = loss_fn(ttft_pred, yb[:, [0]]) + loss_fn(tpot_pred, yb[:, [1]])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1:02d} | Loss={total_loss/len(loader.dataset):.6f}")

# -----------------------------
# 4. Save model + preprocessor
# -----------------------------
os.makedirs("checkpoints/hardware_cost_model", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/hardware_cost_model/model.pt")
print("✅ Model saved to checkpoints/hardware_cost_model/model.pt")
