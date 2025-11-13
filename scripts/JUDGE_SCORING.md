# LLM-as-a-Judge Scoring Guide (vLLM Offline Mode)

This guide explains how to use the `get_scores.py` script to evaluate LLM responses using Qwen2.5-72B-Instruct as a judge in **vLLM offline mode**.

## Overview

The script processes CSV files containing input questions and LLM-generated answers, then:
1. Loads the judge model directly using vLLM (offline mode)
2. Processes prompts in batches for efficient GPU utilization
3. Receives a score (0.0-1.0) and justification in JSON format
4. Adds two new columns (`judge_score` and `judge_justification`) to the CSV
5. Saves progress after each batch and supports resume functionality

**Key Advantage**: Unlike API mode, this loads the model once and processes all files efficiently without needing a separate server.

## Prerequisites

### 1. Install Dependencies

The script requires `vllm`, `pandas`, and `tqdm`:

```bash
# Activate environment
source .venv/bin/activate

# vLLM should already be installed
# If not: pip install vllm pandas tqdm
```

### 2. GPU Requirements

The judge model requires significant GPU memory:

- **Qwen2.5-72B-Instruct**: 4x A100 (80GB) or 8x A100 (40GB) with tensor parallelism
- **Qwen2.5-32B-Instruct**: 2x A100 (80GB) or 4x A100 (40GB) - faster alternative
- **Qwen2.5-14B-Instruct**: 1x A100 (40GB) - smaller alternative

## Usage

### Quick Start (Default Configuration)

Process all three CSV files with default settings:

```bash
# Using the convenience script
bash scripts/run_judge_scoring.sh

# Or directly with Python
python scripts/get_scores.py --model Qwen/Qwen2.5-72B-Instruct
```

This will process:
- `data/Mistral-7B-Instruct-v0.3.csv`
- `data/Phi-3-mini-128k-instruct.csv`
- `data/Qwen2.5-3B-Instruct.csv`

Output files will be saved as `*_scored.csv` in the `data/` directory.

### Custom Configuration

#### Process a Single File

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --input data/Mistral-7B-Instruct-v0.3.csv
```

#### Use Smaller Judge Model

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2
```

#### Adjust Tensor Parallelism

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 8  # Use 8 GPUs
```

#### Adjust Batch Size

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --batch_size 64  # Process 64 samples at once (faster)
```

#### Disable Resume (Start Fresh)

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --no_resume  # Overwrite existing output files
```

#### Custom Cache Directory

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --cache-dir /path/to/your/cache
```

### Full Command Example

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --input data/Mistral-7B-Instruct-v0.3.csv data/Phi-3-mini-128k-instruct.csv \
    --output_suffix _scored \
    --batch_size 64 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --temperature 0.0 \
    --max-tokens 512
```

## Judge Prompt Template

The script uses the following evaluation criteria:

**Score Ranges:**
- **0.0-0.3**: Poor quality - incorrect, unhelpful, or severely incomplete
- **0.4-0.6**: Moderate quality - partially correct but with significant issues
- **0.7-0.8**: Good quality - mostly correct and helpful with minor issues
- **0.9-1.0**: Excellent quality - accurate, comprehensive, and highly helpful

The judge evaluates based on:
- Accuracy
- Helpfulness
- Completeness
- Relevance to the question

## Output Format

The script adds two new columns to your CSV:

| Original Columns | New Columns |
|-----------------|-------------|
| id | judge_score |
| source | judge_justification |
| prompt | |
| p_tokens | |
| input_tokens | |
| output_text | |
| output_tokens | |

**Example output:**

```csv
id,source,prompt,...,judge_score,judge_justification
123,mix_instruct,"Explain quantum computing",0.85,"The response provides a clear and accurate explanation of quantum computing..."
124,mix_instruct,"What is photosynthesis?",0.90,"Excellent comprehensive explanation covering all key aspects..."
```

## Features

### Batch Processing

- Processes multiple samples at once for efficiency
- Default batch size: 32 (adjustable with `--batch_size`)
- Larger batches = faster processing but more GPU memory

### Resume Functionality

The script automatically saves progress after each batch. If interrupted:

1. Restart with the same command
2. It will detect existing output and resume from the last completed batch
3. Already-scored rows are skipped

To start fresh instead of resuming, use `--no_resume`.

### Error Handling

- Robust JSON parsing with fallback mechanisms
- Logs warnings for unparseable responses
- Continues processing even if some batches fail
- Saves progress before crashing

### Progress Tracking

- Real-time progress bar using tqdm
- Batch-level progress updates
- Final statistics (average score, completion rate)

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *Required* | Judge model (e.g., Qwen/Qwen2.5-72B-Instruct) |
| `--input` | All 3 CSV files | Input file(s) to process |
| `--output_suffix` | `_scored` | Suffix for output filenames |
| `--batch_size` | 32 | Samples per batch |
| `--tensor-parallel-size` | 1 | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization` | 0.9 | GPU memory fraction (0.0-1.0) |
| `--cache-dir` | `/blue/sgao1/...` | Model cache directory |
| `--temperature` | 0.0 | Sampling temperature |
| `--max-tokens` | 512 | Max tokens for judge response |
| `--no_resume` | False | Disable resume functionality |

## Troubleshooting

### Out of Memory Errors

**Problem**: CUDA OOM or vLLM initialization fails

**Solutions**:
1. **Increase tensor parallelism**: Use more GPUs
   ```bash
   --tensor-parallel-size 8
   ```
2. **Use smaller model**: Switch to 32B or 14B variant
   ```bash
   --model Qwen/Qwen2.5-32B-Instruct
   ```
3. **Reduce GPU memory utilization**:
   ```bash
   --gpu-memory-utilization 0.8
   ```
4. **Reduce batch size**:
   ```bash
   --batch_size 16
   ```

### Parsing Errors

**Problem**: `Failed to parse judge response`

**Solutions**:
1. Check logs for actual response content
2. Script automatically retries with fallback parsing
3. Consider adjusting temperature or prompt template
4. Some failures are normal; check completion rate at end

### Slow Processing

**Problem**: Scoring takes too long

**Solutions**:
1. **Increase batch size** (if GPU memory allows):
   ```bash
   --batch_size 128
   ```
2. **Use smaller model**:
   ```bash
   --model Qwen/Qwen2.5-14B-Instruct
   ```
3. **Reduce max tokens**:
   ```bash
   --max-tokens 256
   ```
4. **Process files separately** in parallel on different GPU nodes

### Model Download Issues

**Problem**: Model fails to download or load

**Solutions**:
1. Check internet connection and HuggingFace access
2. Pre-download model:
   ```bash
   huggingface-cli download Qwen/Qwen2.5-72B-Instruct
   ```
3. Verify cache directory permissions
4. Set custom cache directory:
   ```bash
   --cache-dir /path/with/space
   ```

## Environment Variables

You can set environment variables for the shell script:

```bash
export JUDGE_MODEL="Qwen/Qwen2.5-32B-Instruct"
export TENSOR_PARALLEL_SIZE=2
export GPU_MEMORY_UTIL=0.85
export BATCH_SIZE=64
bash scripts/run_judge_scoring.sh
```

## Example Session

```bash
$ python scripts/get_scores.py --model Qwen/Qwen2.5-72B-Instruct --batch_size 32

2025-11-12 10:00:00 - INFO - Initializing judge model...
2025-11-12 10:00:01 - INFO - === Initializing vLLM with model: Qwen/Qwen2.5-72B-Instruct ===
2025-11-12 10:00:01 - INFO - Cache directory: /blue/sgao1/ji757406.ucf/hf_cache/
2025-11-12 10:00:01 - INFO - Tensor parallel size: 4
2025-11-12 10:00:01 - INFO - GPU memory utilization: 0.9
INFO 11-12 10:00:05 llm_engine.py:98] Initializing an LLM engine (v0.6.0) with config: ...
2025-11-12 10:00:30 - INFO - === Model initialized successfully ===
2025-11-12 10:00:30 - INFO - Processing data/Mistral-7B-Instruct-v0.3.csv
2025-11-12 10:00:30 - INFO - Loaded 100 rows from data/Mistral-7B-Instruct-v0.3.csv
Scoring Mistral-7B-Instruct-v0.3.csv: 100%|████████| 4/4 [01:23<00:00, 20.8s/batch]
2025-11-12 10:01:53 - INFO - Completed processing data/Mistral-7B-Instruct-v0.3.csv
2025-11-12 10:01:53 - INFO - Results saved to data/Mistral-7B-Instruct-v0.3_scored.csv
2025-11-12 10:01:53 - INFO - Average score: 0.762
2025-11-12 10:01:53 - INFO - Valid scores: 100/100
```

## Performance Tips

### Optimize for Speed

1. **Maximum batch size**: Start with 32, increase to 64/128 if memory allows
2. **Multiple GPUs**: Use tensor parallelism for large models
3. **Process in parallel**: Run multiple instances on different files

### Optimize for Memory

1. **Reduce batch size**: Lower to 8 or 16
2. **Smaller model**: Use 32B or 14B variant
3. **Lower GPU utilization**: Set to 0.7 or 0.8

### Recommended Configurations

**High-speed (8x A100 80GB):**
```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 8 \
    --batch_size 128 \
    --gpu-memory-utilization 0.9
```

**Balanced (4x A100 80GB):**
```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --batch_size 64 \
    --gpu-memory-utilization 0.9
```

**Memory-efficient (2x A100 40GB):**
```bash
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2 \
    --batch_size 32 \
    --gpu-memory-utilization 0.85
```

## Advanced: Modifying the Judge Template

To customize the evaluation criteria, edit the `JUDGE_TEMPLATE` in `get_scores.py` (lines 41-63):

```python
JUDGE_TEMPLATE = """You are an expert AI judge...

Scoring criteria:
- 0.0-0.3: [Your custom criteria]
- 0.4-0.6: [Your custom criteria]
...

**Question/Prompt:**
{prompt}

**AI-Generated Answer:**
{answer}

Your evaluation:"""
```

## Comparison: Offline vs API Mode

| Feature | Offline Mode (Current) | API Mode |
|---------|----------------------|----------|
| Setup | Direct model loading | Requires separate server |
| Performance | Faster (batch processing) | Slower (sequential) |
| GPU Efficiency | High (keeps model loaded) | Medium (per-request overhead) |
| Scalability | Single process | Multiple clients possible |
| Best For | Batch scoring tasks | Interactive/distributed use |

## Support

For issues:
1. Check vLLM logs for model loading errors
2. Review script logs for parsing/processing errors
3. Verify CSV format matches expected columns (`prompt`, `output_text`)
4. Check GPU availability: `nvidia-smi`
