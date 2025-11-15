"""
Precompute CARROT embeddings for all prompts in the eval dataset.

Output:
    mixed_prompts_eval_with_prompt_embeddings.parquet

This file contains:
    - prompt
    - truncated_prompt (<=512 tokens)
    - carrot_emb (embedding vector as list)
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import torch

MAX_TOKENS = 512
ENCODER_MODEL = "all-MiniLM-L6-v2"   # CARROT encoder


def truncate_prompt(text, max_tokens=MAX_TOKENS):
    words = text.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens])
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/prompts/mixed_prompts_eval.parquet")
    parser.add_argument("--output", default="data/prompts/mixed_prompts_eval_with_prompt_embeddings.parquet")
    args = parser.parse_args()

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    print(f"📂 Loading prompts: {args.input}")
    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    print(f"   Loaded {len(df)} prompts")

    # -----------------------------------------------------
    # Truncate prompts
    # -----------------------------------------------------
    print(f"✂️  Truncating prompts to {MAX_TOKENS} tokens")
    df["truncated_prompt"] = df["prompt"].apply(truncate_prompt)

    # -----------------------------------------------------
    # Load GPU encoder
    # -----------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔌 Loading encoder: {ENCODER_MODEL}  (device={device})")

    encoder = SentenceTransformer(ENCODER_MODEL, device=device)

    # -----------------------------------------------------
    # Encode prompts on GPU
    # -----------------------------------------------------
    print("⚡ Encoding prompts on GPU ...")

    embeddings = []
    for p in tqdm(df["truncated_prompt"], desc="Encoding", ncols=80):
        emb = encoder.encode(p, convert_to_numpy=True)
        embeddings.append(emb.tolist())   # save as python list (parquet-friendly)

    df["carrot_emb"] = embeddings

    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------
    print(f"💾 Saving to {args.output}")
    if args.output.endswith(".parquet"):
        df.to_parquet(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)

    # -----------------------------------------------------
    # Show sample rows
    # -----------------------------------------------------
    print("\n================ SAMPLE SAVED ROWS (10) ================")
    sample = df.head(10).copy()
    # Don't print huge embeddings
    sample["carrot_emb"] = sample["carrot_emb"].apply(lambda x: f"[len={len(x)}]")
    print(sample.to_string(index=False))
    print("========================================================\n")

    print("✅ Finished computing + saving embeddings.")


if __name__ == "__main__":
    main()
