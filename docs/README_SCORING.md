# LLM-as-a-Judge Scoring System

A complete system for evaluating LLM responses using a judge model (Qwen2.5-72B-Instruct) via vLLM offline mode.

## Quick Start

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run scoring on all three CSV files
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4

# Or use the convenience script
bash scripts/run_judge_scoring.sh
```

## What It Does

1. **Loads** a judge model (Qwen2.5-72B-Instruct) using vLLM
2. **Reads** CSV files with columns: `prompt`, `output_text`
3. **Evaluates** each response using structured prompts
4. **Generates** scores (0.0-1.0) and justifications
5. **Adds** two new columns: `judge_score`, `judge_justification`
6. **Saves** results to `*_scored.csv` files

## Files Included

### Main Script
- **`get_scores.py`** - The main scoring script (vLLM offline mode)

### Helper Scripts
- **`run_judge_scoring.sh`** - Convenience wrapper script

### Documentation
- **`JUDGE_SCORING.md`** - Complete usage guide with troubleshooting
- **`SCORING_EXAMPLES.md`** - Quick reference examples for common use cases
- **`README_SCORING.md`** - This file (overview)

## Input Files

The script processes these CSV files by default:
- `data/Mistral-7B-Instruct-v0.3.csv`
- `data/Phi-3-mini-128k-instruct.csv`
- `data/Qwen2.5-3B-Instruct.csv`

**Required columns:**
- `prompt` - The input question
- `output_text` - The LLM's response to evaluate

## Output Files

Creates new CSV files with all original columns plus:
- `judge_score` - Float from 0.0 to 1.0
- `judge_justification` - Text explanation

**Output files:**
- `data/Mistral-7B-Instruct-v0.3_scored.csv`
- `data/Phi-3-mini-128k-instruct_scored.csv`
- `data/Qwen2.5-3B-Instruct_scored.csv`

## Scoring Criteria

The judge evaluates responses on:

| Score Range | Quality Level | Description |
|-------------|---------------|-------------|
| 0.0-0.3 | Poor | Incorrect, unhelpful, or severely incomplete |
| 0.4-0.6 | Moderate | Partially correct but with significant issues |
| 0.7-0.8 | Good | Mostly correct and helpful with minor issues |
| 0.9-1.0 | Excellent | Accurate, comprehensive, and highly helpful |

## Key Features

✅ **Batch Processing** - Processes multiple samples at once (default: 32)
✅ **Auto-Resume** - Continues from last saved progress if interrupted
✅ **Efficient** - Loads model once, processes all files
✅ **Robust** - Handles parsing errors, retries, and edge cases
✅ **Configurable** - Supports various models, batch sizes, and GPU configs

## Requirements

### Software
- Python 3.12 (from `.venv`)
- vLLM (already installed)
- pandas (already installed)
- tqdm (already installed)

### Hardware
- **Recommended**: 4x A100 80GB GPUs (for Qwen2.5-72B)
- **Minimum**: 2x A100 80GB GPUs (use Qwen2.5-32B instead)
- **Alternative**: 1x A100 80GB (use Qwen2.5-14B)

## Common Commands

### Basic Usage
```bash
# Score all files
python scripts/get_scores.py --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4

# Score one file
python scripts/get_scores.py --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4 \
    --input data/Mistral-7B-Instruct-v0.3.csv
```

### Different Models
```bash
# Large (72B) - best quality
python scripts/get_scores.py --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4

# Medium (32B) - good balance
python scripts/get_scores.py --model Qwen/Qwen2.5-32B-Instruct --tensor-parallel-size 2

# Small (14B) - fastest
python scripts/get_scores.py --model Qwen/Qwen2.5-14B-Instruct
```

### Performance Tuning
```bash
# Maximum speed
python scripts/get_scores.py --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 8 --batch_size 128

# Memory efficient
python scripts/get_scores.py --model Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2 --batch_size 16 --gpu-memory-utilization 0.8
```

## Documentation

📖 **New to the system?** Start with:
1. This README for overview
2. `JUDGE_SCORING.md` for detailed guide
3. `SCORING_EXAMPLES.md` for quick examples

## Example Output

```csv
id,source,prompt,p_tokens,input_tokens,output_text,output_tokens,judge_score,judge_justification
123,mix_instruct,"Rate this song: We Will Rock You",24,25,"I would rate it 10. This song...",129,0.85,"The response provides a clear rating with strong justification..."
```

## Workflow

```
┌─────────────────┐
│  Input CSV      │
│  (prompt +      │
│   output_text)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load Judge     │
│  Model (vLLM)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Batch Process  │
│  (32 at a time) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Parse JSON     │
│  Extract Score  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save Progress  │
│  After Each     │
│  Batch          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output CSV     │
│  (+ score +     │
│   justification)│
└─────────────────┘
```

## Monitoring Progress

While the script runs:

```bash
# Terminal 1: Run scoring
python scripts/get_scores.py --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4

# Terminal 2: Monitor progress
watch -n 5 'wc -l data/Mistral-7B-Instruct-v0.3_scored.csv'

# Terminal 3: Check GPU usage
watch -n 1 nvidia-smi
```

## Troubleshooting

### Out of Memory
- Use smaller model (32B or 14B)
- Increase `--tensor-parallel-size`
- Reduce `--batch_size`
- Lower `--gpu-memory-utilization`

### Parsing Errors
- Normal to have a few (1-2%)
- Check logs for details
- Script continues processing

### Slow Performance
- Increase `--batch_size` (if memory allows)
- Use more GPUs with `--tensor-parallel-size`
- Process files in parallel (separate jobs)

See `JUDGE_SCORING.md` for detailed troubleshooting.

## Performance Estimates

For ~100 samples per file:

| Configuration | Time per File |
|---------------|---------------|
| 72B, 8 GPUs, batch 128 | ~10 minutes |
| 72B, 4 GPUs, batch 64 | ~20 minutes |
| 32B, 2 GPUs, batch 64 | ~15 minutes |
| 14B, 1 GPU, batch 128 | ~12 minutes |

## Support

For issues or questions:
1. Check `JUDGE_SCORING.md` for detailed documentation
2. Review `SCORING_EXAMPLES.md` for usage examples
3. Check logs for error messages
4. Verify GPU availability: `nvidia-smi`

## Architecture

The system uses **vLLM offline mode**:
- Loads model directly (no separate server needed)
- Efficient batch processing
- Automatic memory management
- Supports tensor parallelism for large models

**Advantages over API mode:**
- ✅ Simpler setup (no server required)
- ✅ Faster (batch processing)
- ✅ Better GPU utilization
- ✅ Single command workflow

## Credits

Based on the vLLM inference pattern from `scripts/extract.py`.

Judge template adapted from standard LLM-as-a-Judge evaluation practices.
