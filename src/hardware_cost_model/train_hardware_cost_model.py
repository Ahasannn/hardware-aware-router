"""
train_hardware_cost_model.py
Train Hardware Cost Model to predict TTFT & TPOT.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump
from .model_utils import HardwareCostNet


# -----------------------------
# 1. Load and preprocess dataset
# -----------------------------
CSV_PATH = "data/hw_dataset_qwen_sweep.csv"
df = pd.read_csv(CSV_PATH)

# Drop irrelevant columns
df = df.drop(columns=["request_id", "timestamp", "prompt_id", "latency_s", "e2e_avg"], errors="ignore")

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

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Preprocessing done | Features={X.shape[1]} | Train={len(X_train)} | Val={len(X_val)}")

# -----------------------------
# 2. Training setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

def to_loader(X, y, batch=256, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch, shuffle=shuffle)

train_loader = to_loader(X_train, y_train)
val_loader = to_loader(X_val, y_val, shuffle=False)

model = HardwareCostNet(X.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# -----------------------------
# 3. Training loop
# -----------------------------
for epoch in range(25):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        ttft_pred, tpot_pred = model(xb)
        loss = loss_fn(ttft_pred, yb[:, [0]]) + loss_fn(tpot_pred, yb[:, [1]])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / len(train_loader.dataset)

    # ---- validation ----
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            ttft_pred, tpot_pred = model(xb)
            val_loss += (loss_fn(ttft_pred, yb[:, [0]]) + loss_fn(tpot_pred, yb[:, [1]])).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1:02d} | Train={train_loss:.6f} | Val={val_loss:.6f}")

# -----------------------------
# 4. Save model + preprocessor
# -----------------------------
os.makedirs("checkpoints/hardware_cost_model", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/hardware_cost_model/model.pt")
print("✅ Model saved to checkpoints/hardware_cost_model/model.pt")
