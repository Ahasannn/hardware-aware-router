"""
combine_mixinstruct_longbench.py
Combine MixInstruct and LongBench datasets,
shuffle them, and save as a single mixed dataset.
"""

import os
import pandas as pd
from sklearn.utils import shuffle

# --------------- CONFIG ----------------
MIXINSTRUCT_PATH = "src/data/mixinstruct_prompts.parquet"
LONGBENCH_PATH = "src/data/longbench_prompts.parquet"
OUT_PATH = "src/data/mixed_prompts_final.parquet"
RANDOM_SEED = 42
# ---------------------------------------

def combine_datasets():
    print("📂 Loading MixInstruct dataset ...")
    mix_df = pd.read_parquet(MIXINSTRUCT_PATH)
    print(f"   ✅ Loaded {len(mix_df)} MixInstruct samples.")

    print("📂 Loading LongBench dataset ...")
    lb_df = pd.read_parquet(LONGBENCH_PATH)
    print(f"   ✅ Loaded {len(lb_df)} LongBench samples.")

    # --- Standardize columns ---
    required_cols = ["id", "source", "prompt", "p_tokens"]
    mix_df = mix_df[required_cols]
    lb_df = lb_df[required_cols]

    # --- Combine + shuffle ---
    combined_df = pd.concat([mix_df, lb_df], ignore_index=True)
    combined_df = shuffle(combined_df, random_state=RANDOM_SEED).reset_index(drop=True)

    # --- Save ---
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    combined_df.to_parquet(OUT_PATH, index=False)

    print(f"✅ Combined dataset saved → {OUT_PATH}")
    print(f"   Total samples: {len(combined_df)}")
    print("   Composition:")
    print(combined_df['source'].value_counts())

    return combined_df


if __name__ == "__main__":
    df = combine_datasets()
    print(df.head())
