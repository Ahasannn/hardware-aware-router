#!/usr/bin/env python3
"""
LLM-as-a-Judge scoring script using vLLM offline mode.

This script evaluates LLM responses using Qwen2.5-72B-Instruct as a judge.
It processes CSV files containing input questions and output answers, generates
scores (0.0-1.0) and justifications, and adds them as new columns.

Usage:
    python scripts/get_scores.py --model Qwen/Qwen2.5-72B-Instruct --input data/Mistral-7B-Instruct-v0.3.csv
    python scripts/get_scores.py --model Qwen/Qwen2.5-72B-Instruct  # Process all default files
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Tuple, Optional, List

import pandas as pd
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("ERROR: vLLM not installed. Install with: pip install vllm")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Judge prompt template
JUDGE_TEMPLATE = """You are an expert AI judge tasked with evaluating the quality of AI-generated responses. You will be given a question/prompt and an AI-generated answer. Your job is to:

1. Assess the quality, accuracy, helpfulness, and completeness of the response
2. Provide a score from 0.0 to 1.0 (in increments of 0.1)
3. Provide a brief justification for your score

Scoring criteria:
- 0.0-0.3: Poor quality - incorrect, unhelpful, or severely incomplete
- 0.4-0.6: Moderate quality - partially correct but with significant issues
- 0.7-0.8: Good quality - mostly correct and helpful with minor issues
- 0.9-1.0: Excellent quality - accurate, comprehensive, and highly helpful

**Question/Prompt:**
{prompt}

**AI-Generated Answer:**
{answer}

**Instructions:**
Respond with ONLY a JSON object in the following format (no other text):
{{"score": <float between 0.0 and 1.0>, "justification": "<your brief explanation>"}}

Your evaluation:"""


def parse_judge_response(response: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Parse the judge's response to extract score and justification.

    Args:
        response: Raw text response from the judge model

    Returns:
        Tuple of (score, justification, json_response). Returns (None, None, None) if parsing fails.
    """
    try:
        # Try to find JSON in the response
        # Look for {...} pattern
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)

            score = data.get('score')
            justification = data.get('justification', '')

            # Validate score
            if score is not None:
                score = float(score)
                if 0.0 <= score <= 1.0:
                    return score, justification, json_str
                else:
                    logger.warning(f"Score {score} out of valid range [0.0, 1.0]")

        # If JSON parsing fails, try to extract score from text
        score_match = re.search(r'score["\s:]+([0-9.]+)', response, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            if 0.0 <= score <= 1.0:
                return score, response.strip(), None

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse judge response: {e}")
        logger.debug(f"Response was: {response[:200]}")

    return None, None, None


def initialize_judge_model(
    model_name: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    cache_dir: str = None,
    max_model_len: int = 8192
) -> LLM:
    """
    Initialize the vLLM judge model.

    Args:
        model_name: HuggingFace model name or path
        tensor_parallel_size: Number of GPUs to use
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        cache_dir: Directory to cache downloaded models
        max_model_len: Maximum context length (default: 8192)

    Returns:
        Initialized LLM instance
    """
    # Set up cache directory
    if cache_dir is None:
        cache_dir = "/blue/sgao1/ji757406.ucf/hf_cache/"

    os.makedirs(cache_dir, exist_ok=True)

    # Set HuggingFace cache environment variable
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir

    logger.info(f"=== Initializing vLLM with model: {model_name} ===")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
    logger.info(f"Max model length: {max_model_len}")

    # Initialize vLLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        download_dir=cache_dir,
        max_model_len=max_model_len,
    )

    logger.info("=== Model initialized successfully ===")
    return llm


def get_judge_scores_batch(
    llm: LLM,
    prompts: List[str],
    answers: List[str],
    temperature: float = 0.0,
    max_tokens: int = 512
) -> List[Tuple[Optional[float], Optional[str], Optional[str]]]:
    """
    Get scores and justifications from the judge model for a batch of prompts.

    Args:
        llm: Initialized vLLM instance
        prompts: List of input questions/prompts
        answers: List of AI-generated answers to evaluate
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        List of tuples (score, justification, json_response)
    """
    # Prepare judge prompts
    judge_prompts = [
        JUDGE_TEMPLATE.format(prompt=p, answer=a)
        for p, a in zip(prompts, answers)
    ]

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=None,
    )

    # Run inference
    outputs = llm.generate(judge_prompts, sampling_params)

    # Parse results
    results = []
    for output in outputs:
        judge_output = output.outputs[0].text
        score, justification, json_response = parse_judge_response(judge_output)
        results.append((score, justification, json_response))

    return results


def process_csv_file(
    input_path: Path,
    output_path: Path,
    llm: LLM,
    batch_size: int = 32,
    temperature: float = 0.0,
    max_tokens: int = 512,
    resume: bool = True
) -> None:
    """
    Process a CSV file and add judge scores and justifications.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        llm: Initialized vLLM judge model
        batch_size: Number of samples to process at once
        temperature: Sampling temperature
        max_tokens: Maximum tokens for judge response
        resume: If True, resume from existing output file
    """
    logger.info(f"Processing {input_path}")

    # Read input CSV
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")

    # Check if we should resume
    start_idx = 0
    if resume and output_path.exists():
        existing_df = pd.read_csv(output_path)
        if 'judge_score' in existing_df.columns:
            # Find last valid score
            valid_scores = existing_df['judge_score'].notna()
            if valid_scores.any():
                start_idx = valid_scores[::-1].idxmax() + 1
                logger.info(f"Resuming from row {start_idx}")
                df = existing_df

    # Add columns if they don't exist
    if 'judge_score' not in df.columns:
        df['judge_score'] = None
    if 'judge_justification' not in df.columns:
        df['judge_justification'] = None
    if 'judge_json' not in df.columns:
        df['judge_json'] = None

    # Process in batches
    try:
        total_batches = (len(df) - start_idx + batch_size - 1) // batch_size

        for batch_start in tqdm(
            range(start_idx, len(df), batch_size),
            desc=f"Scoring {input_path.name}",
            total=total_batches
        ):
            batch_end = min(batch_start + batch_size, len(df))
            batch_indices = range(batch_start, batch_end)

            # Get batch data
            batch_prompts = []
            batch_answers = []
            valid_indices = []

            for idx in batch_indices:
                row = df.iloc[idx]
                # Skip if already scored
                if pd.notna(row.get('judge_score')):
                    continue

                batch_prompts.append(row['prompt'])
                batch_answers.append(row['output_text'])
                valid_indices.append(idx)

            # Process batch if there are unscored items
            if batch_prompts:
                results = get_judge_scores_batch(
                    llm=llm,
                    prompts=batch_prompts,
                    answers=batch_answers,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # Update dataframe
                for idx, (score, justification, json_response) in zip(valid_indices, results):
                    df.at[idx, 'judge_score'] = score
                    df.at[idx, 'judge_justification'] = justification
                    df.at[idx, 'judge_json'] = json_response

                    if score is None:
                        logger.warning(f"Row {idx}: Failed to get valid score for ID {df.at[idx, 'id']}")

            # Save progress after each batch
            df.to_csv(output_path, index=False)

        # Final save
        df.to_csv(output_path, index=False)
        logger.info(f"Completed processing {input_path}")
        logger.info(f"Results saved to {output_path}")

        # Print statistics
        valid_scores = df['judge_score'].notna()
        if valid_scores.any():
            mean_score = df.loc[valid_scores, 'judge_score'].mean()
            logger.info(f"Average score: {mean_score:.3f}")
            logger.info(f"Valid scores: {valid_scores.sum()}/{len(df)}")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Saving progress...")
        df.to_csv(output_path, index=False)
        logger.info(f"Progress saved to {output_path}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Score LLM responses using an LLM-as-a-Judge via vLLM offline mode"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Judge model name (e.g., Qwen/Qwen2.5-72B-Instruct)'
    )
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        default=[
            'data/Mistral-7B-Instruct-v0.3.csv',
            'data/Phi-3-mini-128k-instruct.csv',
            'data/Qwen2.5-3B-Instruct.csv'
        ],
        help='Input CSV file(s) to process'
    )
    parser.add_argument(
        '--output_suffix',
        type=str,
        default='_scored',
        help='Suffix to add to output filenames (default: _scored)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Number of samples to process at once (default: 32)'
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='Do not resume from existing output files'
    )
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism (default: 1)'
    )
    parser.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.9,
        help='GPU memory utilization fraction (default: 0.9)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='/blue/sgao1/ji757406.ucf/hf_cache/',
        help='Directory to cache downloaded models'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature for judge (default: 0.0)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens for judge response (default: 512)'
    )
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=8192,
        help='Maximum context length for the model (default: 8192, reduce if OOM)'
    )

    args = parser.parse_args()

    # Initialize the judge model once
    logger.info("Initializing judge model...")
    llm = initialize_judge_model(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cache_dir=args.cache_dir,
        max_model_len=args.max_model_len
    )

    # Process each input file
    for input_file in args.input:
        input_path = Path(input_file)

        if not input_path.exists():
            logger.warning(f"Input file not found: {input_path}")
            continue

        # Generate output path
        output_path = input_path.parent / f"{input_path.stem}{args.output_suffix}{input_path.suffix}"

        try:
            process_csv_file(
                input_path=input_path,
                output_path=output_path,
                llm=llm,
                batch_size=args.batch_size,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                resume=not args.no_resume
            )
        except Exception as e:
            logger.error(f"Failed to process {input_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("All files processed successfully!")


if __name__ == '__main__':
    main()
