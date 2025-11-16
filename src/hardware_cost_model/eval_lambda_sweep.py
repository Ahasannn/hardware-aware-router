"""
eval_lambda_sweep.py

Updated version:
- CARROT cost = static CARROT token pricing only
- OUR cost    = pure latency-based cost only
- Completely separate cost definitions
- No mixing between CARROT cost and latency cost

SLO definitions:
- SLO(TTFT) = a + b * p_tokens (prefill scaling model)
- SLO(TPOT) = P70 percentile
- SLO(E2E)  = TTFT_SLO + TPOT_SLO * d_tokens
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------
#  SLO Calibration
# ---------------------------------------------------------

def fit_ttft_slo(df):
    valid = df.dropna(subset=["ttft_s", "p_tokens"])
    X = valid["p_tokens"].values.reshape(-1, 1)
    y = valid["ttft_s"].values

    model = LinearRegression()
    model.fit(X, y)

    a = float(model.intercept_)
    b = float(model.coef_[0])

    slack = 1.20
    return a * slack, b * slack


def compute_tpot_slo(df):
    vals = df["tpot_s_per_token"].dropna()
    return float(np.percentile(vals, 70))


# ---------------------------------------------------------
#  λ-sweep Logic
# ---------------------------------------------------------

def run_lambda_sweep(csv_path, lambdas=None):
    print(f"[Sweep] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Predicted latency = TTFT + len * TPOT
    df["pred_total_latency"] = (
        df["predicted_ttft"] +
        df["carrot_predicted_length"] * df["predicted_tpot"]
    )

    # Clean up invalid values
    df["pred_total_latency"] = (
        df["pred_total_latency"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(df["pred_total_latency"].median())
        .clip(lower=1e-6)
    )

    # -------------------------------
    # COST DEFINITIONS
    # -------------------------------

    # CARROT COST — keep exactly as CARROT uses it (normalized)
    max_static = df["carrot_predicted_cost"].max()
    df["static_cost_norm"] = df["carrot_predicted_cost"] / (max_static + 1e-9)

    # OUR COST — pure latency cost
    raw_lat = df["pred_total_latency"].copy()
    raw_lat = raw_lat.replace([np.inf, -np.inf], np.nan).fillna(raw_lat.median())
    raw_lat = raw_lat.clip(lower=1e-6)

    max_log_lat = np.log1p(raw_lat).max()
    df["latency_cost_norm"] = np.log1p(raw_lat) / (max_log_lat + 1e-9)


    # ---------------------------------------------------------
    # SLO definitions
    # ---------------------------------------------------------
    slo_a, slo_b = fit_ttft_slo(df)
    slo_tpot = compute_tpot_slo(df)

    print("\nCalibrated SLO definitions:")
    print(f"  TTFT_SLO(p) = {slo_a:.3f} + {slo_b:.7f} * p_tokens")
    print(f"  TPOT_SLO    = {slo_tpot:.5f} s/token\n")

    groups = df.groupby("prompt_id")

    # ---------------------------------------------------------
    # λ sweep
    # ---------------------------------------------------------

    for lam in lambdas:

        # -----------------------------------------
        # 1. CARROT Score
        # -----------------------------------------
        df["carrot_score"] = (
            lam * df["carrot_predicted_quality"]
            - (1 - lam) * df["static_cost_norm"]
        )

        # -----------------------------------------
        # 2. OUR Score
        # -----------------------------------------
        df["ours_score"] = (
            lam * df["carrot_predicted_quality"]
            - (1 - lam) * df["latency_cost_norm"]
        )

        # Best per prompt
        idx_c = groups["carrot_score"].idxmax()
        idx_o = groups["ours_score"].idxmax()

        sel_c = df.loc[idx_c].copy()
        sel_o = df.loc[idx_o].copy()

        # -----------------------------------------
        # Quality and latency stats
        # -----------------------------------------
        carrot_q = sel_c["actual_quality_score"].mean()
        carrot_lat = sel_c["latency_s"].mean()

        ours_q = sel_o["actual_quality_score"].mean()
        ours_lat = sel_o["latency_s"].mean()

        # -----------------------------------------
        # SLO metrics
        # -----------------------------------------
        # TTFT SLO
        c_ttft_slo = slo_a + slo_b * sel_c["p_tokens"]
        o_ttft_slo = slo_a + slo_b * sel_o["p_tokens"]

        carrot_slo_ttft = (sel_c["ttft_s"] <= c_ttft_slo).mean()
        ours_slo_ttft = (sel_o["ttft_s"] <= o_ttft_slo).mean()

        # TPOT SLO
        carrot_slo_tpot = (sel_c["tpot_s_per_token"] <= slo_tpot).mean()
        ours_slo_tpot = (sel_o["tpot_s_per_token"] <= slo_tpot).mean()

        # E2E SLO
        c_e2e_slo = c_ttft_slo + slo_tpot * sel_c["d_tokens"]
        o_e2e_slo = o_ttft_slo + slo_tpot * sel_o["d_tokens"]

        carrot_slo_e2e = (sel_c["latency_s"] <= c_e2e_slo).mean()
        ours_slo_e2e = (sel_o["latency_s"] <= o_e2e_slo).mean()

        # -----------------------------------------
        # Pretty print
        # -----------------------------------------
        print(f"\nλ = {lam:.2f}")
        print(f"  CARROT: Q={carrot_q:.4f}  LAT={carrot_lat:.2f}s  | "
              f"SLO(TTFT)={carrot_slo_ttft:.3f}  SLO(TPOT)={carrot_slo_tpot:.3f}  SLO(E2E)={carrot_slo_e2e:.3f}")

        print(f"  OURS  : Q={ours_q:.4f}  LAT={ours_lat:.2f}s  | "
              f"SLO(TTFT)={ours_slo_ttft:.3f}  SLO(TPOT)={ours_slo_tpot:.3f}  SLO(E2E)={ours_slo_e2e:.3f}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
        default="data/evaluation_dataset_processed_full.csv")
    parser.add_argument("--lambda_start", type=float, default=0.0)
    parser.add_argument("--lambda_end",   type=float, default=1.0)
    parser.add_argument("--lambda_step",  type=float, default=0.1)

    args = parser.parse_args()

    lambdas = np.arange(
        args.lambda_start,
        args.lambda_end + 1e-9,
        args.lambda_step
    ).tolist()

    run_lambda_sweep(args.input, lambdas)


if __name__ == "__main__":
    main()
