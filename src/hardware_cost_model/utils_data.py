import os, datetime, torch, pandas as pd

def now_iso():
    return datetime.datetime.now().isoformat()

def safe_gpu_name(gid: int):
    try:
        return torch.cuda.get_device_name(gid)
    except Exception:
        return f"GPU-{gid}"

def token_len(prompt: str) -> int:
    return max(1, len(prompt.split()))

def load_local_prompts(parquet_path: str, n: int, seed: int = 42):
    df = pd.read_parquet(parquet_path)
    n = min(n, len(df))
    sampled = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return [(str(i), row["prompt"]) for i, row in sampled.iterrows()]
