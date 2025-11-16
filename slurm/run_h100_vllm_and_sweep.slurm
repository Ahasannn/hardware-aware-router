#!/bin/bash
#SBATCH --job-name=vllm_pool_run
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# ==========================================================
# Environment setup
# ==========================================================
source ~/hardware-aware-router/.venv/bin/activate
export HF_HOME=/lustre/fs1/home/$USER/.cache/huggingface
export HF_TOKEN=REDACTED_TOKEN    # optional if models are public
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "🧭 Node: $(hostname)"
echo "🧩 GPUs visible: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# ==========================================================
# Kill any residual vLLM servers
# ==========================================================
echo "🔪 Killing existing vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 100

echo "🚀 Starting vLLM servers on 2×H100..."
# ==========================================================
# GPU 0 models
# ==========================================================
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen14b \
  --max-model-len 21000 \
  --gpu-memory-utilization 0.55 \
  --port 8010 \
  --host 0.0.0.0 \
  > logs/gpu0_qwen2.5_14b.log 2>&1 &

sleep 300

CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/phi3-mini \
  --max-model-len 21000 \
  --gpu-memory-utilization 0.3 \
  --port 8011 \
  --host 0.0.0.0 \
  > logs/gpu0_phi3_mini.log 2>&1 &

sleep 300

# ==========================================================
# GPU 1 models
# ==========================================================

CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/llama3-8b \
  --max-model-len 21000 \
  --gpu-memory-utilization 0.4 \
  --port 8012 \
  --host 0.0.0.0 \
  > logs/gpu1-llama3-8b.log 2>&1 &

 sleep 300

CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen3b \
  --max-model-len 21000 \
  --gpu-memory-utilization 0.20 \
  --port 8013 \
  --host 0.0.0.0 \
  > logs/gpu1-qwen3b.log 2>&1 &

sleep 300


CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/mistral7b \
  --max-model-len 21000 \
  --gpu-memory-utilization 0.25 \
  --port 8014 \
  --host 0.0.0.0 \
  > logs/gpu1_mistral7b.log 2>&1 &


sleep 300
echo "✅ vLLM servers should now be ready."
nvidia-smi

# ==========================================================
# Run the sweep workload
# ==========================================================
echo "🎯 Starting hardware cost dataset sweep..."
CONFIG="configs/gpu_model_map_h100.yaml"
OUTPUT="data/hw_dataset_newton_h100.csv"
NUM_PROMPTS=8000
INTERVAL=0.5
CONCURRENCY=16

# Poisson – low/mid/high
for RATE in 2 5 10; do
  LOGFILE="logs/run_poisson_${RATE}.log"
  echo ">>> Running pattern=poisson, rate=${RATE}"
  nohup python -m src.hardware_cost_model.build_hardware_cost_dataset \
    --config ${CONFIG} \
    --output ${OUTPUT} \
    --num_prompts ${NUM_PROMPTS} \
    --interval ${INTERVAL} \
    --pattern poisson \
    --rate ${RATE} \
    --concurrency ${CONCURRENCY} \
    > ${LOGFILE} 2>&1
  echo "✅ Completed poisson rate=${RATE}"
done

# Microburst
LOGFILE="logs/run_microburst_5.log"
echo ">>> Running pattern=microburst, rate=5"
nohup python -m src.hardware_cost_model.build_hardware_cost_dataset \
  --config ${CONFIG} \
  --output ${OUTPUT} \
  --num_prompts ${NUM_PROMPTS} \
  --interval ${INTERVAL} \
  --pattern microburst \
  --rate 5 \
  --concurrency ${CONCURRENCY} \
  > ${LOGFILE} 2>&1
echo "✅ Completed microburst (rate=5)"

# Sustained
LOGFILE="logs/run_sustained_10.log"
echo ">>> Running pattern=sustained, rate=10"
nohup python -m src.hardware_cost_model.build_hardware_cost_dataset \
  --config ${CONFIG} \
  --output ${OUTPUT} \
  --num_prompts ${NUM_PROMPTS} \
  --interval ${INTERVAL} \
  --pattern sustained \
  --rate 10 \
  --concurrency ${CONCURRENCY} \
  > ${LOGFILE} 2>&1
echo "✅ Completed sustained (rate=10)"

echo "🎯 Sweep complete. Dataset appended to: ${OUTPUT}"
echo "----------------------------------------------------------"
echo "📊 Check logs in ./logs/ and GPU usage with nvidia-smi"
echo "----------------------------------------------------------"
