#!/bin/bash
#SBATCH --job-name=vllm_v100_sweep
#SBATCH --partition=normal
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

# ==========================================================
# Environment setup
# ==========================================================

# Detect environment (Newton vs. local lab)
if [ -d "/lustre/fs1/home/$USER" ]; then
    # Running on Newton
    source ~/hardware-aware-router/.venv/bin/activate
    export HF_HOME=/lustre/fs1/home/$USER/.cache/huggingface
    export TRANSFORMERS_CACHE=$HF_HOME/transformers
else
    # Running on local lab GPU
    export HF_HOME=$HOME/.cache/huggingface
    export TRANSFORMERS_CACHE=$HF_HOME/transformers
fi

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
sleep 3

echo "🚀 Starting vLLM servers on 2×V100..."
# ==========================================================
# GPU 0 models
# ==========================================================
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen1.5-0.5B \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.4 \
  --port 8005 --host 0.0.0.0 \
  > logs/qwen0_0.5b.log 2>&1 &

sleep 10

CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.3 \
  --port 8006 --host 0.0.0.0 \
  > logs/qwen0_1.5b.log 2>&1 &

sleep 10

# ==========================================================
# GPU 1 models
# ==========================================================
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.3 \
  --port 8007 --host 0.0.0.0 \
  > logs/qwen1_0.5B-Chat-GPTQ-Int4.log 2>&1 &

sleep 10

CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct-AWQ \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.45 \
  --port 8008 --host 0.0.0.0 \
  > logs/qwen1_1.7b.log 2>&1 &

sleep 10

echo "⏳ Waiting 4 minutes for models to finish loading..."

echo "✅ vLLM servers should now be ready."
nvidia-smi

# ==========================================================
# Run the sweep workload
# ==========================================================
echo "🎯 Starting hardware cost dataset sweep..."
CONFIG="configs/gpu_model_map_qwen.yaml"
OUTPUT="data/hw_dataset_newton.csv"
NUM_PROMPTS=10
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
