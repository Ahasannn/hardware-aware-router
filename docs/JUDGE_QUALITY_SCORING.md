# LLM-as-a-Judge Quality Scoring

This guide explains how to score LLM responses using a judge model (Qwen2.5-72B-Instruct) via vLLM. The resulting scores are used to build the training data for IRT and UMR baselines.

## Quick Start

```bash
# Score all five model output files (uses 4 GPUs by default)
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4

# Or use the convenience wrapper
bash scripts/run_judge_scoring.sh
```

By default, this processes all five per-model CSVs in `data/data_quality/`:
- `Llama-3.1-8B-Instruct.csv`
- `Mistral-7B-Instruct-v0.3.csv`
- `Phi-3-mini-128k-instruct.csv`
- `Qwen2.5-14B-Instruct.csv`
- `Qwen2.5-3B-Instruct.csv`

Each file gets a corresponding `*_scored.csv` output with two new columns: `judge_score` (0.0–1.0) and `judge_justification`.

## GPU Requirements

| Judge Model | Minimum GPUs | Recommended |
|-------------|-------------|-------------|
| Qwen2.5-72B | 4× A100 40GB | 4× A100 80GB |
| Qwen2.5-32B | 2× A100 40GB | 2× A100 80GB |
| Qwen2.5-14B | 1× A100 40GB | 1× A100 80GB |

## Common Commands

```bash
# Score a single model's output file
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --input data/Mistral-7B-Instruct-v0.3.csv

# Use a smaller/faster judge
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2

# Start fresh (don't resume previous run)
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --no_resume
```

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *required* | Judge model (HuggingFace ID) |
| `--input` | All 3 CSVs | Input file(s) to process |
| `--output_suffix` | `_scored` | Suffix for output filenames |
| `--batch_size` | 32 | Samples per batch |
| `--tensor-parallel-size` | 1 | Number of GPUs |
| `--gpu-memory-utilization` | 0.9 | GPU memory fraction |
| `--temperature` | 0.0 | Sampling temperature (0.0 = deterministic) |
| `--max-tokens` | 512 | Max tokens in judge response |
| `--no_resume` | False | Disable auto-resume |

## Scoring Criteria

| Score | Quality |
|-------|---------|
| 0.0–0.3 | Poor — incorrect, unhelpful, or severely incomplete |
| 0.4–0.6 | Moderate — partially correct with significant issues |
| 0.7–0.8 | Good — mostly correct with minor issues |
| 0.9–1.0 | Excellent — accurate, comprehensive, highly helpful |

## SLURM / HPC Usage

```bash
# Single node, 4 GPUs
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=judge_scoring
#SBATCH --gres=gpu:a100:4
#SBATCH --time=04:00:00

python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --batch_size 64
EOF

# Parallel jobs — one per model file
for model in Mistral-7B-Instruct-v0.3 Phi-3-mini-128k-instruct Qwen2.5-3B-Instruct; do
    sbatch --job-name=score_${model} --gres=gpu:a100:4 --wrap \
        "python scripts/get_scores.py --model Qwen/Qwen2.5-72B-Instruct \
         --tensor-parallel-size 4 --input data/${model}.csv"
done
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA OOM | Increase `--tensor-parallel-size`, use 32B model, or reduce `--batch_size` |
| Slow processing | Increase `--batch_size`, use more GPUs, or process files in parallel |
| Parsing errors | Normal at 1–2% rate; script continues and logs failures |
| Model download fails | Pre-download: `huggingface-cli download Qwen/Qwen2.5-72B-Instruct` |

## Resume After Interruption

The script saves progress after each batch. To resume, just rerun the same command — it automatically detects and skips already-scored rows.
