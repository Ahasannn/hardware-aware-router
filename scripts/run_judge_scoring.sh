#!/bin/bash
# Shell script to run LLM-as-a-Judge scoring on all CSV files using vLLM offline mode

# Configuration
MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "=========================================="
echo "LLM-as-a-Judge Scoring Script (Offline Mode)"
echo "=========================================="
echo "Judge Model: $MODEL"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo "Batch Size: $BATCH_SIZE"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "=========================================="
echo ""

# Activate environment if needed
if [ -d ".venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run the scoring script
python scripts/get_scores.py \
    --model "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --batch_size "$BATCH_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --input \
        data/Mistral-7B-Instruct-v0.3.csv \
        data/Phi-3-mini-128k-instruct.csv \
        data/Qwen2.5-3B-Instruct.csv

echo ""
echo "=========================================="
echo "Scoring complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - data/Mistral-7B-Instruct-v0.3_scored.csv"
echo "  - data/Phi-3-mini-128k-instruct_scored.csv"
echo "  - data/Qwen2.5-3B-Instruct_scored.csv"
echo ""
