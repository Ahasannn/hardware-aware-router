# Running with Qwen3-Next-80B-A3B-Instruct

This guide provides specific instructions for using the Qwen3-Next-80B-A3B-Instruct model as the judge.

## The Problem You Encountered

**Error:** `CUDA out of memory. Tried to allocate 2.00 GiB... Using max model len 262144`

**Root Cause:** The Qwen3-Next-80B model supports up to 256K tokens context length. vLLM automatically allocates memory for the full KV cache, which requires enormous GPU memory (262,144 tokens × 80B parameters = massive memory).

**Solution:** Limit the context length to what you actually need for judging (8K tokens is more than enough).

## Recommended Configurations

### Configuration 1: Multiple A100 80GB GPUs (Recommended)

For best performance with Qwen3-Next-80B:

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --batch_size 32
```

**Requirements:**
- 4x A100 80GB GPUs
- Max context: 8K tokens (sufficient for judge prompts)
- GPU memory: 85% utilization

### Configuration 2: More GPUs (Safer)

If you have 8 GPUs available:

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --batch_size 64
```

**Benefits:**
- More headroom
- Faster inference
- Larger batch sizes possible

### Configuration 3: Minimal Memory (4K Context)

If still getting OOM with 8K:

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --batch_size 16
```

**Trade-offs:**
- Shorter context (4K still plenty for judging)
- Smaller batches (slower)
- Lower memory usage

### Configuration 4: Using Smaller Qwen Model (Alternative)

If you don't need the 80B model, use a smaller variant:

```bash
# Qwen2.5-72B (much more memory efficient)
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --batch_size 64

# Qwen2.5-32B (even smaller)
python scripts/get_scores.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --batch_size 128
```

## Why 8192 Context Length?

For LLM-as-a-Judge tasks, you need:
- Judge template: ~200 tokens
- Input prompt: ~50-500 tokens (your dataset has p_tokens column)
- Output to evaluate: ~100-500 tokens (your output_text)
- Judge response: ~200-500 tokens

**Total:** ~550-1700 tokens per sample

**8192 tokens** gives you plenty of headroom (~5x buffer) while using **32x less memory** than the default 262K.

## Understanding max_model_len Impact on Memory

| max_model_len | KV Cache Memory (80B model) | Relative Memory |
|---------------|----------------------------|-----------------|
| 262144 (auto) | ~200 GB | 32x |
| 32768 | ~25 GB | 4x |
| 16384 | ~12 GB | 2x |
| 8192 | ~6 GB | 1x (baseline) |
| 4096 | ~3 GB | 0.5x |

*Approximate values for 80B parameter model with bf16*

## Step-by-Step: First Time Running

### 1. Check Available GPUs

```bash
nvidia-smi
```

Look for:
- Number of GPUs available
- Memory per GPU (80GB, 40GB, etc.)
- Current memory usage

### 2. Start with Conservative Settings

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --batch_size 8 \
    --input data/Mistral-7B-Instruct-v0.3.csv
```

### 3. Monitor During Loading

```bash
# In another terminal
watch -n 1 nvidia-smi
```

### 4. If It Loads Successfully

Gradually increase:
1. `--max-model-len 8192` (double context)
2. `--batch_size 16` (double batch)
3. `--gpu-memory-utilization 0.85` (more memory)
4. `--batch_size 32` (double batch again)

### 5. If Still Getting OOM

Try:
1. Increase `--tensor-parallel-size 8` (use more GPUs)
2. Reduce `--max-model-len 2048` (shorter context)
3. Keep `--batch_size 4` (very small batches)

## Using Environment Variables

```bash
# Set your preferences
export JUDGE_MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
export TENSOR_PARALLEL_SIZE=4
export MAX_MODEL_LEN=8192
export GPU_MEMORY_UTIL=0.85
export BATCH_SIZE=32

# Run
bash scripts/run_judge_scoring.sh
```

## Comparison: Qwen3-Next vs Qwen2.5

| Model | Size | Context | Quality | Speed | Memory |
|-------|------|---------|---------|-------|--------|
| Qwen3-Next-80B | 80B | 256K | Highest | Slowest | Highest |
| Qwen2.5-72B | 72B | 128K | Very High | Fast | High |
| Qwen2.5-32B | 32B | 128K | High | Very Fast | Medium |
| Qwen2.5-14B | 14B | 128K | Good | Fastest | Low |

**Recommendation for Judge Tasks:** Qwen2.5-72B offers the best balance of quality and efficiency.

## Troubleshooting

### Error: "CUDA out of memory" during model loading

**Solutions:**
1. ✅ **Add `--max-model-len 8192`** (most important!)
2. Increase `--tensor-parallel-size` (use more GPUs)
3. Reduce `--gpu-memory-utilization 0.8`
4. Try smaller model (Qwen2.5-72B)

### Error: "CUDA out of memory" during inference

**Solutions:**
1. Reduce `--batch_size 16` (or 8, 4)
2. Reduce `--max-model-len 4096`
3. Lower `--gpu-memory-utilization 0.75`

### Warning: "Using max model len 262144"

**This is the problem!** Add `--max-model-len 8192` to fix it.

### Model loads but inference is very slow

**Solutions:**
1. Increase `--batch_size` (32 → 64 → 128)
2. Increase `--tensor-parallel-size` (use more GPUs)
3. Use smaller model for faster processing

## Example: Full Command with All Parameters

```bash
python scripts/get_scores.py \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --input data/Mistral-7B-Instruct-v0.3.csv \
    --output_suffix _scored_qwen3 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --batch_size 32 \
    --temperature 0.0 \
    --max-tokens 512 \
    --cache-dir ~/.cache/vllm/
```

## Performance Expectations

With Qwen3-Next-80B-A3B (4x A100 80GB, batch_size=32, max_model_len=8192):

- **Model loading:** ~2-3 minutes
- **Inference speed:** ~100-150 samples/hour
- **Total time (100 samples):** ~40-60 minutes per file
- **Total time (3 files):** ~2-3 hours

## When to Use Qwen3-Next-80B

✅ **Use if:**
- You need absolute best quality judgments
- You have 4+ A100 80GB GPUs available
- Time is not critical
- You're willing to wait for superior results

❌ **Consider alternatives if:**
- You have limited GPUs (<4 A100s)
- You need faster turnaround
- Quality difference vs 72B is not critical
- You're getting persistent OOM errors

## Quick Decision Guide

**You have:**
- **8x A100 80GB** → Use Qwen3-Next-80B with TP=8, batch=64
- **4x A100 80GB** → Use Qwen3-Next-80B with TP=4, batch=32, OR Qwen2.5-72B with TP=4
- **2x A100 80GB** → Use Qwen2.5-72B with TP=2 or Qwen2.5-32B with TP=2
- **1x A100 80GB** → Use Qwen2.5-32B with TP=1 or Qwen2.5-14B

**Priority:**
1. **Quality First** → Qwen3-Next-80B or Qwen2.5-72B
2. **Speed First** → Qwen2.5-32B or Qwen2.5-14B
3. **Memory Constrained** → Start with smallest model that meets quality needs
