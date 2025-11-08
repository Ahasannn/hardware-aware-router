"""
model_utils.py
Reusable definitions for Hardware Cost Model.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from joblib import load


# -----------------------------
# 1. Model Definition
# -----------------------------
class HardwareCostNet(nn.Module):
    """
    3-layer shared MLP with two heads:
      Input → Linear(→128) → GELU → Linear(→64) → GELU
      Heads: TTFT (64→1), TPOT (64→1)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.ttft_head = nn.Linear(64, 1)
        self.tpot_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.ttft_head(h), self.tpot_head(h)


# -----------------------------
# 2. Load Model + Preprocessor
# -----------------------------
def load_cost_model(ckpt_dir="checkpoints/hardware_cost_model"):
    """Load trained cost model and its preprocessor for runtime inference."""
    preproc_path = os.path.join(ckpt_dir, "preproc.joblib")
    model_path = os.path.join(ckpt_dir, "model.pt")

    # Load preprocessor
    preproc = load(preproc_path)

    # Build model with correct input dimension
    input_dim = len(preproc.get_feature_names_out())
    model = HardwareCostNet(input_dim)

    # Load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    print(f"✅ Loaded Hardware Cost Model from: {model_path}")
    return preproc, model, device


# -----------------------------
# 3. Inference Function
# -----------------------------
def predict_ttft_tpot(preproc, model, features_dict, device):
    """Predict TTFT & TPOT given a single request's hardware + prompt features."""
    model_gpu = f"{features_dict['model_id']}_{features_dict['gpu_id']}"
    df_in = pd.DataFrame([{
        "p_tokens": features_dict["p_tokens"],
        "running_req_count": features_dict["running_req_count"],
        "waiting_req_count": features_dict["waiting_req_count"],
        "kv_cache_usage_perc": features_dict["kv_cache_usage_perc"],
        "ttft_avg": features_dict["ttft_avg"],
        "itl_avg": features_dict["itl_avg"],
        "model_gpu": model_gpu
    }])
    x = preproc.transform(df_in)
    x_t = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        ttft_hat, tpot_hat = model(x_t)
    return float(torch.exp(ttft_hat)), float(torch.exp(tpot_hat))
