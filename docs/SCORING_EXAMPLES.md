# LLM-as-a-Judge Scoring Examples

Quick reference examples for common use cases.

## Basic Usage

### 1. Score All Three Files (Default)

```bash
# Using 4 GPUs with Qwen2.5-72B
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4
```

**Output:**
- `data/Mistral-7B-Instruct-v0.3_scored.csv`
- `data/Phi-3-mini-128k-instruct_scored.csv`
- `data/Qwen2.5-3B-Instruct_scored.csv`

### 2. Score a Single File

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --input data/Mistral-7B-Instruct-v0.3.csv
```

### 3. Using Convenience Script

```bash
# Set environment variables
export JUDGE_MODEL="Qwen/Qwen2.5-72B-Instruct"
export TENSOR_PARALLEL_SIZE=4

# Run
bash scripts/run_judge_scoring.sh
```

## Different Model Sizes

### Large Model (72B) - Best Quality

```bash
# Requires 4x A100 80GB
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --batch_size 32
```

### Medium Model (32B) - Good Balance

```bash
# Requires 2x A100 80GB
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2 \
    --batch_size 64
```

### Small Model (14B) - Fast

```bash
# Requires 1x A100 80GB
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --batch_size 128
```

## Performance Optimization

### Maximum Speed (8 GPUs)

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 8 \
    --batch_size 128 \
    --gpu-memory-utilization 0.95
```

**Estimated throughput:** ~600 samples/hour

### Memory-Efficient (Limited GPUs)

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2 \
    --batch_size 16 \
    --gpu-memory-utilization 0.8
```

## Processing Subsets

### Process Only One Model's Results

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --input data/Mistral-7B-Instruct-v0.3.csv
```

### Process Multiple Specific Files

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --input \
        data/Mistral-7B-Instruct-v0.3.csv \
        data/Phi-3-mini-128k-instruct.csv
```

## Custom Output

### Custom Output Suffix

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --output_suffix _judge_72b
```

**Output:** `data/Mistral-7B-Instruct-v0.3_judge_72b.csv`

### Custom Cache Directory

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --cache-dir /scratch/my_cache
```

## Resume & Recovery

### Resume After Interruption

```bash
# Just rerun the same command - it will auto-resume
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4
```

### Start Fresh (Overwrite)

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --no_resume
```

## Slurm/HPC Examples

### Single Node, Multiple GPUs

```bash
#!/bin/bash
#SBATCH --job-name=judge_scoring
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --time=04:00:00
#SBATCH --mem=256G

source .venv/bin/activate

python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --batch_size 64
```

### Process Files in Parallel (Multiple Jobs)

```bash
# Job 1: Score Mistral
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=score_mistral
#SBATCH --gres=gpu:a100:4
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --input data/Mistral-7B-Instruct-v0.3.csv
EOF

# Job 2: Score Phi-3
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=score_phi3
#SBATCH --gres=gpu:a100:4
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --input data/Phi-3-mini-128k-instruct.csv
EOF

# Job 3: Score Qwen
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=score_qwen
#SBATCH --gres=gpu:a100:4
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --input data/Qwen2.5-3B-Instruct.csv
EOF
```

## Temperature & Creativity

### Deterministic (Default)

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --temperature 0.0
```

**Best for:** Consistent, reproducible scores

### Slightly Creative

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --temperature 0.3
```

**Best for:** More varied justifications

## Monitoring Progress

### Check Output File During Processing

```bash
# In another terminal
watch -n 5 'wc -l data/Mistral-7B-Instruct-v0.3_scored.csv'
```

### View Latest Scores

```bash
# Show last 5 scored entries
tail -n 5 data/Mistral-7B-Instruct-v0.3_scored.csv | column -t -s ','
```

### Calculate Average Score So Far

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/Mistral-7B-Instruct-v0.3_scored.csv')
print(f'Scored: {df[\"judge_score\"].notna().sum()}/{len(df)}')
print(f'Average: {df[\"judge_score\"].mean():.3f}')
"
```

## Analyzing Results

### Compare Model Scores

```python
import pandas as pd

# Load all scored files
mistral = pd.read_csv('data/Mistral-7B-Instruct-v0.3_scored.csv')
phi3 = pd.read_csv('data/Phi-3-mini-128k-instruct_scored.csv')
qwen = pd.read_csv('data/Qwen2.5-3B-Instruct_scored.csv')

# Compare average scores
print(f"Mistral-7B: {mistral['judge_score'].mean():.3f}")
print(f"Phi-3-mini: {phi3['judge_score'].mean():.3f}")
print(f"Qwen2.5-3B: {qwen['judge_score'].mean():.3f}")

# Distribution
print("\nMistral-7B Score Distribution:")
print(mistral['judge_score'].value_counts(bins=10, sort=False))
```

### Export High-Quality Responses

```python
import pandas as pd

df = pd.read_csv('data/Mistral-7B-Instruct-v0.3_scored.csv')

# Get high-scoring responses (>= 0.8)
high_quality = df[df['judge_score'] >= 0.8]
high_quality.to_csv('data/high_quality_responses.csv', index=False)

print(f"Found {len(high_quality)} high-quality responses")
```

## Troubleshooting Examples

### Test With Small Sample

```bash
# Create test subset
head -n 11 data/Mistral-7B-Instruct-v0.3.csv > data/test_sample.csv

# Score it
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --input data/test_sample.csv
```

### Debug Mode (Verbose Logging)

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --batch_size 1 \
    2>&1 | tee judge_scoring.log
```

### Check GPU Usage

```bash
# In another terminal while scoring runs
watch -n 1 nvidia-smi
```

## Quick Reference Table

| Use Case | Model | GPUs | Batch Size | Speed |
|----------|-------|------|------------|-------|
| Best Quality | 72B | 4-8 | 32-64 | Medium |
| Balanced | 32B | 2-4 | 64 | Fast |
| Fast Testing | 14B | 1-2 | 128 | Very Fast |
| Memory Constrained | 32B | 2 | 16 | Slow |
| Maximum Speed | 72B | 8 | 128 | Very Fast |

## Estimated Processing Times

Assuming ~100 samples per file:

| Configuration | Time per File | Total Time (3 files) |
|---------------|---------------|---------------------|
| 72B, 8 GPUs, batch 128 | ~10 min | ~30 min |
| 72B, 4 GPUs, batch 64 | ~20 min | ~60 min |
| 32B, 2 GPUs, batch 64 | ~15 min | ~45 min |
| 14B, 1 GPU, batch 128 | ~12 min | ~36 min |

*Note: Actual times vary based on prompt lengths and hardware*
