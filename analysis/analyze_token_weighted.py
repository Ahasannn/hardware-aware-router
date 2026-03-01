#!/usr/bin/env python3
"""
Analysis script to compare baseline hw router vs token-weighted router.
Compares P95 latency improvements.
"""

import pandas as pd
import numpy as np
import glob
import sys
from pathlib import Path


def find_latest_csv(pattern):
    """Find the most recent CSV matching pattern"""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=lambda f: Path(f).stat().st_mtime)


def analyze_router_results(baseline_csv, token_weighted_csv):
    """Compare baseline vs token-weighted router results"""

    # Load results
    print(f"Loading baseline: {baseline_csv}")
    baseline = pd.read_csv(baseline_csv)

    print(f"Loading token-weighted: {token_weighted_csv}")
    token_weighted = pd.read_csv(token_weighted_csv)

    # Validate data
    if "latency_real" not in baseline.columns or "latency_real" not in token_weighted.columns:
        print("ERROR: Missing 'latency_real' column in results")
        return

    # Calculate metrics
    print("\n" + "="*60)
    print("LATENCY COMPARISON (seconds)")
    print("="*60)

    # P50, P95, P99 latencies
    for percentile, label in [(0.5, "P50"), (0.95, "P95"), (0.99, "P99")]:
        baseline_val = baseline["latency_real"].quantile(percentile)
        token_weighted_val = token_weighted["latency_real"].quantile(percentile)
        improvement = (baseline_val - token_weighted_val) / baseline_val * 100

        print(f"\n{label} Latency:")
        print(f"  Baseline:       {baseline_val:.3f}s")
        print(f"  Token-weighted: {token_weighted_val:.3f}s")
        print(f"  Improvement:    {improvement:.2f}%")

    # Mean latency
    baseline_mean = baseline["latency_real"].mean()
    token_weighted_mean = token_weighted["latency_real"].mean()
    mean_improvement = (baseline_mean - token_weighted_mean) / baseline_mean * 100

    print(f"\nMean Latency:")
    print(f"  Baseline:       {baseline_mean:.3f}s")
    print(f"  Token-weighted: {token_weighted_mean:.3f}s")
    print(f"  Improvement:    {mean_improvement:.2f}%")

    # Request counts
    print(f"\nRequests:")
    print(f"  Baseline:       {len(baseline)}")
    print(f"  Token-weighted: {len(token_weighted)}")

    # Model distribution
    print("\n" + "="*60)
    print("MODEL ROUTING DISTRIBUTION")
    print("="*60)

    if "model_hf" in baseline.columns and "model_hf" in token_weighted.columns:
        print("\nBaseline model distribution:")
        print(baseline["model_hf"].value_counts())

        print("\nToken-weighted model distribution:")
        print(token_weighted["model_hf"].value_counts())

    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze token-weighted experiment results")
    parser.add_argument("--baseline", type=str,
                        help="Path to baseline router results CSV (or glob pattern)")
    parser.add_argument("--token_weighted", type=str,
                        help="Path to token-weighted router results CSV (or glob pattern)")
    parser.add_argument("--output_dir", type=str,
                        default="data/token_weighted_experiment",
                        help="Directory containing experiment results")

    args = parser.parse_args()

    # Auto-find CSVs if not specified
    if not args.baseline:
        args.baseline = find_latest_csv(f"{args.output_dir}/*_hw_router_results.csv")
    if not args.token_weighted:
        args.token_weighted = find_latest_csv(f"{args.output_dir}/*_hw_token_weighted_router_results.csv")

    if not args.baseline or not args.token_weighted:
        print("ERROR: Could not find result CSVs")
        print(f"Baseline: {args.baseline}")
        print(f"Token-weighted: {args.token_weighted}")
        sys.exit(1)

    analyze_router_results(args.baseline, args.token_weighted)
