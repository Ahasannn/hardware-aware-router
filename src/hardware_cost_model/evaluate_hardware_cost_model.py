"""
evaluate_hardware_cost_model.py
Evaluate trained Hardware Cost Model on held-out data and generate paper-ready figures/tables.
"""

import os, torch, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .model_utils import HardwareCostNet

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

# same cleanup as training
df = df.drop(columns=["request_id", "timestamp", "prompt_id", "latency_s", "e2e_avg"], errors="ignore")
df["model_gpu"] = df["model_id"].astype(str) + "_" + df["gpu_id"].astype(str)
df = df.drop(columns=["gpu_id"], errors="ignore")

df["ttft_s"] = df["ttft_s"].clip(lower=1e-4)
df["tpot_s_per_token"] = df["tpot_s_per_token"].clip(lower=1e-4)
df["ttft_s_log"] = np.log(df["ttft_s"])
df["tpot_s_log"] = np.log(df["tpot_s_per_token"])

preproc = load(PREPROC_PATH)
X = preproc.transform(df)
y = df[["ttft_s_log", "tpot_s_log"]].values

# reproducible split
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Get correct transformed feature names
try:
    feature_names = preproc.get_feature_names_out()
except AttributeError:
    feature_names = [f"feat_{i}" for i in range(X_val.shape[1])]

feature_df = pd.DataFrame(X_val, columns=feature_names)

# -----------------------------
# 2. Load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HardwareCostNet(X.shape[1]).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# 3. Inference
# -----------------------------
with torch.no_grad():
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    ttft_pred, tpot_pred = model(Xv)
    y_pred = np.column_stack([ttft_pred.cpu().numpy().flatten(),
                              tpot_pred.cpu().numpy().flatten()])

# -----------------------------
# 4. Metrics
# -----------------------------
def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

log_ttft = metrics(y_val[:, 0], y_pred[:, 0])
log_tpot = metrics(y_val[:, 1], y_pred[:, 1])

# Back-transform
ttft_true, ttft_pred_lin = np.exp(y_val[:, 0]), np.exp(y_pred[:, 0])
tpot_true, tpot_pred_lin = np.exp(y_val[:, 1]), np.exp(y_pred[:, 1])
lin_ttft = metrics(ttft_true, ttft_pred_lin)
lin_tpot = metrics(tpot_true, tpot_pred_lin)

# Save metrics table
metrics_df = pd.DataFrame([
    ["TTFT (log)", *log_ttft.values()],
    ["TPOT (log)", *log_tpot.values()],
    ["TTFT (orig)", *lin_ttft.values()],
    ["TPOT (orig)", *lin_tpot.values()],
], columns=["Metric", "MAE", "RMSE", "R2"])
metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_table.csv"), index=False)
print(metrics_df)

# -----------------------------
# 5. Visualization
# -----------------------------
def scatter_plot(y_true, y_pred, title, fname):
    plt.figure()
    sns.scatterplot(x=y_true, y=y_pred, s=10, alpha=0.4)
    lim = [0, max(y_true.max(), y_pred.max())]
    plt.plot(lim, lim, 'r--')
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

def feature_corr_plot(feature, y_true, y_pred, fname):
    plt.figure()
    plt.scatter(feature, np.abs(y_true - y_pred), s=8, alpha=0.3)
    plt.xlabel(feature.name)
    plt.ylabel("|Error|")
    plt.title(f"Error vs {feature.name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=300)
    plt.close()

# Scatter & error plots
scatter_plot(ttft_true, ttft_pred_lin, "TTFT Prediction (Original Scale)", "ttft_true_vs_pred.png")
scatter_plot(tpot_true, tpot_pred_lin, "TPOT Prediction (Original Scale)", "tpot_true_vs_pred.png")
error_dist(ttft_true, ttft_pred_lin, "TTFT Error Distribution", "ttft_error_dist.png")
error_dist(tpot_true, tpot_pred_lin, "TPOT Error Distribution", "tpot_error_dist.png")

# Feature sensitivity
if "num__p_tokens" in feature_df.columns:
    feature_corr_plot(feature_df["num__p_tokens"], ttft_true, ttft_pred_lin, "ttft_error_vs_p_tokens.png")
if "num__kv_cache_usage_perc" in feature_df.columns:
    feature_corr_plot(feature_df["num__kv_cache_usage_perc"], ttft_true, ttft_pred_lin, "ttft_error_vs_kv_cache.png")

# -----------------------------
# 6. Sample predictions
# -----------------------------
sample_df = pd.DataFrame({
    "ttft_true_s": ttft_true,
    "ttft_pred_s": ttft_pred_lin,
    "tpot_true_s_per_tok": tpot_true,
    "tpot_pred_s_per_tok": tpot_pred_lin,
})
sample_df["ttft_error_pct"] = 100 * (sample_df["ttft_pred_s"] - sample_df["ttft_true_s"]) / sample_df["ttft_true_s"]
sample_df["tpot_error_pct"] = 100 * (sample_df["tpot_pred_s_per_tok"] - sample_df["tpot_true_s_per_tok"]) / sample_df["tpot_true_s_per_tok"]
sample_df.head(50).to_csv(os.path.join(OUT_DIR, "sample_predictions.csv"), index=False)

print(f"\n✅ All figures and tables saved in {OUT_DIR}")
