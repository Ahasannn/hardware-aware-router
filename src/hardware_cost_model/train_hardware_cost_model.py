"""
train_hardware_cost_model.py
Train Hardware Cost Model to predict TTFT & TPOT.
"""

import os
import sys
import time
import platform
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
from .model_maps import (
    get_model_id,
    get_model_hugging_face_name
)

# Parse command line arguments
FORCE_CPU = "--cpu" in sys.argv or "--force-cpu" in sys.argv


# -----------------------------
# 0. System Information
# -----------------------------
print("=" * 80)
print("SYSTEM CONFIGURATION")
print("=" * 80)
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CPU Cores: {os.cpu_count()}")
print("=" * 80)
print()

# Start total timer
total_start_time = time.time()

# -----------------------------
# 1. Load and preprocess dataset
# -----------------------------
print("📊 Loading dataset...")
load_start = time.time()
CSV_PATH = "data/h100_full_sweep.csv"
df = pd.read_csv(CSV_PATH)
load_time = time.time() - load_start
print(f"✅ Dataset loaded in {load_time:.2f}s | Shape: {df.shape}")

print("🔧 Preprocessing data...")
preproc_start = time.time()

# Drop irrelevant columns
df = df.drop(columns=["request_id", "timestamp", "prompt_id", "latency_s", "e2e_avg"], errors="ignore")

# Combine model_id + gpu_id
df["model_id"] = df["model_id"].map(get_model_id)
df["model_gpu"] = df["model_id"].astype(int).astype(str) + "_" + df["gpu_id"].astype(str)
df = df.drop(columns=["gpu_id"], errors="ignore")

if df["model_id"].isna().any():
    missing = df[df["model_id"].isna()]["model_id"].unique()
    raise ValueError(f"❌ Unknown model names in CSV: {missing}")

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
preproc_time = time.time() - preproc_start
print(f"✅ Preprocessing done in {preproc_time:.2f}s")
print(f"   • Total samples: {len(df):,}")
print(f"   • Training samples: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
print(f"   • Validation samples: {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)")
print(f"   • Features: {X.shape[1]}")
print(f"   • Target variables: TTFT (log) + TPOT (log)")
print()

# -----------------------------
# 2. Training setup
# -----------------------------
if FORCE_CPU:
    device = torch.device("cpu")
    print(f"🧠 Using device: {device} (forced via --cpu flag)")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

def to_loader(X, y, batch=1024, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch, shuffle=shuffle)

train_loader = to_loader(X_train, y_train)
val_loader = to_loader(X_val, y_val, shuffle=False)

model = HardwareCostNet(X.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# -----------------------------
# 3. Training loop
# -----------------------------
print("🚀 Starting training...")
training_start = time.time()

for epoch in range(50):
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

training_time = time.time() - training_start
print(f"\n✅ Training completed in {training_time:.2f}s ({training_time/50:.2f}s per epoch)")

# -----------------------------
# 4. Save model + preprocessor
# -----------------------------
print("\n💾 Saving model...")
save_start = time.time()
os.makedirs("checkpoints/hardware_cost_model", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/hardware_cost_model/model.pt")
save_time = time.time() - save_start
print(f"✅ Model saved to checkpoints/hardware_cost_model/model.pt in {save_time:.2f}s")

# -----------------------------
# 5. Summary
# -----------------------------
total_time = time.time() - total_start_time
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print(f"Device:                {device}")
print(f"Total dataset size:    {len(df):,} samples")
print(f"Training samples:      {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Validation samples:    {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"Input features:        {X.shape[1]}")
print(f"Output targets:        2 (TTFT + TPOT)")
print(f"Training epochs:       50")
print(f"Total iterations:      {len(train_loader) * 50:,} ({len(train_loader)} batches × 50 epochs)")
print("-" * 80)
print(f"Data loading time:     {load_time:>8.2f}s  ({load_time/total_time*100:>5.1f}%)")
print(f"Preprocessing time:    {preproc_time:>8.2f}s  ({preproc_time/total_time*100:>5.1f}%)")
print(f"Training time:         {training_time:>8.2f}s  ({training_time/total_time*100:>5.1f}%)")
print(f"   • Per epoch:        {training_time/50:>8.2f}s")
print(f"   • Per batch:        {training_time/(len(train_loader)*50)*1000:>8.2f}ms")
print(f"Model saving time:     {save_time:>8.2f}s  ({save_time/total_time*100:>5.1f}%)")
print("-" * 80)
print(f"TOTAL TIME:            {total_time:>8.2f}s  ({total_time/60:.2f} minutes)")
print(f"Throughput:            {len(X_train)*50/training_time:>8,.0f} samples/second")
print("=" * 80)
print(f"\n✨ Training complete! The model is lightweight and trains efficiently on CPU.")
print(f"   Trained on {len(df):,} samples in {total_time/60:.1f} minutes.")
