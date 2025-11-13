#!/usr/bin/env python3
"""
Extract and analyze data from mixed_prompts parquet files.
Supports LLM inference using vLLM.
"""
import argparse
import json
from pathlib import Path
from typing import Optional, List

import pandas as pd
from tqdm import tqdm

def load_data(file_path: str) -> pd.DataFrame:
    """Load parquet file into DataFrame."""
    return pd.read_parquet(file_path)


def get_statistics(df: pd.DataFrame) -> dict:
    """Get basic statistics about the dataset."""
    stats = {
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "token_stats": {
            "min": int(df['p_tokens'].min()),
            "max": int(df['p_tokens'].max()),
            "mean": float(df['p_tokens'].mean()),
            "median": float(df['p_tokens'].median()),
        },
        "sources": df['source'].value_counts().to_dict(),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
    }
    return stats


def filter_by_source(df: pd.DataFrame, sources: List[str]) -> pd.DataFrame:
    """Filter DataFrame by source(s)."""
    return df[df['source'].isin(sources)]


def filter_by_tokens(df: pd.DataFrame, min_tokens: Optional[int] = None,
                     max_tokens: Optional[int] = None) -> pd.DataFrame:
    """Filter DataFrame by token count range."""
    result = df.copy()
    if min_tokens is not None:
        result = result[result['p_tokens'] >= min_tokens]
    if max_tokens is not None:
        result = result[result['p_tokens'] <= max_tokens]
    return result


def export_to_json(df: pd.DataFrame, output_path: str,
                   fields: Optional[List[str]] = None):
    """Export DataFrame to JSON file."""
    if fields:
        df = df[fields]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient='records', indent=2)
    print(f"Exported {len(df)} records to {output_path}")


def export_to_csv(df: pd.DataFrame, output_path: str,
                  fields: Optional[List[str]] = None):
    """Export DataFrame to CSV file."""
    if fields:
        df = df[fields]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} records to {output_path}")


def export_to_jsonl(df: pd.DataFrame, output_path: str,
                    fields: Optional[List[str]] = None):
    """Export DataFrame to JSONL (JSON Lines) file."""
    if fields:
        df = df[fields]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient='records', lines=True)
    print(f"Exported {len(df)} records to {output_path}")


def sample_data(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample n random rows from DataFrame."""
    return df.sample(n=min(n, len(df)), random_state=seed)


def run_inference(df: pd.DataFrame, model_name: str,
                  max_tokens: int = 512,
                  temperature: float = 0.7,
                  tensor_parallel_size: int = 1,
                  gpu_memory_utilization: float = 0.9,
                  batch_size: int = 32,
                  cache_dir: str = None) -> pd.DataFrame:
    """
    Run vLLM inference on prompts in the DataFrame.

    Args:
        df: DataFrame with 'prompt' column
        model_name: HuggingFace model name or path
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        tensor_parallel_size: Number of GPUs to use
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        batch_size: Batch size for inference
        cache_dir: Directory to cache downloaded models

    Returns:
        DataFrame with added columns: input_tokens, output_text, output_tokens
    """
    import os
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vLLM not installed. Install with: pip install vllm")

    # Set up cache directory
    if cache_dir is None:
        cache_dir = "/blue/sgao1/ji757406.ucf/hf_cache/"

    os.makedirs(cache_dir, exist_ok=True)

    # Set HuggingFace cache environment variable
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir

    print(f"\n=== Initializing vLLM with model: {model_name} ===")
    print(f"Cache directory: {cache_dir}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")

    # Initialize vLLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        download_dir=cache_dir,
    )

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=None,
    )

    print(f"\n=== Running inference on {len(df)} prompts ===")
    print(f"Max tokens: {max_tokens}, Temperature: {temperature}")

    # Prepare prompts
    prompts = df['prompt'].tolist()

    # Run inference in batches with progress bar
    all_outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Inference"):
        batch_prompts = prompts[i:i + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        all_outputs.extend(outputs)

    # Extract results
    results = []
    for output in all_outputs:
        generated_text = output.outputs[0].text
        input_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)

        results.append({
            'input_tokens': input_tokens,
            'output_text': generated_text,
            'output_tokens': output_tokens,
        })

    # Create results DataFrame and merge with original
    results_df = pd.DataFrame(results)
    result = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    print(f"\n=== Inference complete ===")
    print(f"Average input tokens: {results_df['input_tokens'].mean():.2f}")
    print(f"Average output tokens: {results_df['output_tokens'].mean():.2f}")
    print(f"Total output tokens: {results_df['output_tokens'].sum()}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze mixed_prompts parquet data with optional LLM inference"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="src/data/mixed_prompts_train.parquet",
        help="Input parquet file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (format inferred from extension)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics"
    )
    parser.add_argument(
        "--source",
        type=str,
        nargs="+",
        help="Filter by source(s)"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        help="Minimum number of tokens"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum number of tokens"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Sample N random rows"
    )
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        help="Fields to export (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--head",
        type=int,
        help="Show first N rows"
    )

    # Inference arguments
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run LLM inference on prompts"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model name or path for inference (e.g., meta-llama/Llama-2-7b-hf)"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction (default: 0.9)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/blue/sgao1/ji757406.ucf/hf_cache/",
        help="Directory to cache downloaded models (default: /blue/sgao1/ji757406.ucf/hf_cache/)"
    )

    args = parser.parse_args()

    # Validate inference arguments
    if args.inference and not args.model:
        parser.error("--inference requires --model to be specified")

    # Load data
    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    print(f"Loaded {len(df)} rows")

    # Apply filters
    if args.source:
        df = filter_by_source(df, args.source)
        print(f"Filtered by source(s) {args.source}: {len(df)} rows remaining")

    if args.min_tokens or args.max_tokens:
        df = filter_by_tokens(df, args.min_tokens, args.max_tokens)
        print(f"Filtered by tokens [{args.min_tokens or 0}, {args.max_tokens or 'inf'}]: {len(df)} rows remaining")

    # Sample
    if args.sample:
        df = sample_data(df, args.sample, args.seed)
        print(f"Sampled {len(df)} rows")

    # Run inference if requested
    if args.inference:
        df = run_inference(
            df,
            model_name=args.model,
            max_tokens=args.max_output_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
        )

    # Show statistics
    if args.stats:
        stats = get_statistics(df)
        print("\n=== Dataset Statistics ===")
        print(json.dumps(stats, indent=2))

    # Show head
    if args.head:
        print(f"\n=== First {args.head} rows ===")
        print(df.head(args.head).to_string())

    # Export
    if args.output:
        output_path = Path(args.output)
        ext = output_path.suffix.lower()

        if ext == '.json':
            export_to_json(df, args.output, args.fields)
        elif ext == '.csv':
            export_to_csv(df, args.output, args.fields)
        elif ext == '.jsonl':
            export_to_jsonl(df, args.output, args.fields)
        elif ext == '.parquet':
            if args.fields:
                df = df[args.fields]
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(args.output, index=False)
            print(f"Exported {len(df)} records to {args.output}")
        else:
            print(f"Unsupported output format: {ext}")
            print("Supported formats: .json, .csv, .jsonl, .parquet")

    # If no action specified, show info
    if not any([args.stats, args.output, args.head]):
        print("\n=== Quick Info ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nSources distribution:")
        print(df['source'].value_counts())
        print(f"\nToken statistics:")
        print(df['p_tokens'].describe())


if __name__ == "__main__":
    main()
