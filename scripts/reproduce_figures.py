#!/usr/bin/env python3
"""
Reproduce key figures from the HW-Router paper (DAC 2026).

No GPU required — this script uses pre-collected evaluation data
included in the repository to regenerate the lambda sweep results
and paper figures.

Usage:
    python scripts/reproduce_figures.py
    # or: make reproduce
"""

import os
import sys
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIGURE_DIR = os.path.join(DATA_DIR, "figures")

EVAL_CSV = os.path.join(DATA_DIR, "evaluation_dataset_processed_full_with_umr_irt.csv")
SWEEP_CSV = os.path.join(DATA_DIR, "lambda_sweep_results_final_with_irt_umr.csv")


def check_data():
    """Verify that required data files exist."""
    if not os.path.exists(EVAL_CSV):
        print(f"ERROR: Evaluation data not found at {EVAL_CSV}")
        print("This file should be included in the repository.")
        print("If missing, follow the full pipeline in pipeline/README.md to generate it.")
        sys.exit(1)
    print(f"[OK] Found evaluation data: {EVAL_CSV}")


def run_lambda_sweep():
    """Run the offline lambda sweep evaluation."""
    print("\n--- Running Lambda Sweep ---")
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "pipeline", "evaluation", "eval_lambda_sweep.py"),
        "--input", EVAL_CSV,
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    print(f"[OK] Lambda sweep results saved to: {SWEEP_CSV}")


def generate_figures():
    """Generate paper figures from lambda sweep results."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("\n--- Generating Figures ---")

    plots_dir = os.path.join(REPO_ROOT, "analysis", "plots")

    # Try to generate comparison figure
    comparison_script = os.path.join(plots_dir, "plot_comparison.py")
    if os.path.exists(comparison_script):
        try:
            subprocess.run(
                [sys.executable, comparison_script],
                check=True,
                cwd=plots_dir,
            )
            print("[OK] Comparison figure generated")
        except subprocess.CalledProcessError:
            print("[WARN] Could not generate comparison figure (matplotlib may be missing)")
            print("       Install visualization deps: pip install -e '.[viz]'")

    # Try to generate motivation figure
    motivation_script = os.path.join(plots_dir, "plot_combined_motivation.py")
    if os.path.exists(motivation_script):
        try:
            subprocess.run(
                [sys.executable, motivation_script],
                check=True,
                cwd=plots_dir,
            )
            print("[OK] Motivation figure generated")
        except subprocess.CalledProcessError:
            print("[WARN] Could not generate motivation figure")


def print_summary():
    """Print a summary of the key results."""
    try:
        import pandas as pd
    except ImportError:
        return

    if not os.path.exists(SWEEP_CSV):
        return

    df = pd.read_csv(SWEEP_CSV)
    row = df[df["lambda"].between(0.49, 0.51)].iloc[0]

    print("\n" + "=" * 65)
    print("  HW-Router Key Results (lambda = 0.5)")
    print("=" * 65)
    print(f"  {'Router':<12} {'Quality':>10} {'Latency (s)':>14} {'SLO E2E':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*14} {'-'*10}")
    print(f"  {'CARROT':<12} {row['carrot_quality']:>10.3f} {row['carrot_latency']:>14.1f} {row['carrot_slo_e2e']:>9.1%}")
    print(f"  {'IRT':<12} {row['irt_quality']:>10.3f} {row['irt_latency']:>14.1f} {row['irt_slo_e2e']:>9.1%}")
    print(f"  {'UMR':<12} {row['umr_quality']:>10.3f} {row['umr_latency']:>14.1f} {row['umr_slo_e2e']:>9.1%}")
    print(f"  {'HW-Router':<12} {row['ours_quality']:>10.3f} {row['ours_latency']:>14.1f} {row['ours_slo_e2e']:>9.1%}")
    print("=" * 65)

    speedup = row["carrot_latency"] / row["ours_latency"]
    slo_gap = row["ours_slo_e2e"] - row["carrot_slo_e2e"]
    print(f"\n  Speedup vs CARROT: {speedup:.1f}x")
    print(f"  SLO improvement:  +{slo_gap:.1%}")


if __name__ == "__main__":
    check_data()
    run_lambda_sweep()
    generate_figures()
    print_summary()
    print("\nDone. Figures saved to data/figures/")
