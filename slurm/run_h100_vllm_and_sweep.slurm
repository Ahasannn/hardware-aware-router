#!/bin/bash
#SBATCH --job-name=vllm_pool_run
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# ==========================================================
# Environment setup
# ==========================================================
source ~/hardware-aware-router/.venv/bin/activate

echo "🧭 Node: $(hostname)"
echo "🧩 GPUs visible: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# ==========================================================
# Kill any residual vLLM servers
# ==========================================================
echo "🔪 Killing existing vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 300

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


CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/llama3-8b \
  --max-model-len 21000 \
  --gpu-memory-utilization 0.4 \
  --port 8012 \
  --host 0.0.0.0 \
  > logs/gpu1-llama3-8b.log 2>&1 &

sleep 700

CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/phi3-mini \
  --max-model-len 21000 \
  --gpu-memory-utilization 0.3 \
  --port 8011 \
  --host 0.0.0.0 \
  > logs/gpu0_phi3_mini.log 2>&1 &



# ==========================================================
# GPU 1 models
# ==========================================================


CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen3b \
  --max-model-len 21000 \
  --gpu-memory-utilization 0.20 \
  --port 8013 \
  --host 0.0.0.0 \
  > logs/gpu1-qwen3b.log 2>&1 &

sleep 700


CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/mistral7b \
  --max-model-len 21000 \
  --gpu-memory-utilization 0.30 \
  --port 8014 \
  --host 0.0.0.0 \
  > logs/gpu1_mistral7b.log 2>&1 &


sleep 300
echo "✅ vLLM servers should now be ready."
nvidia-smi

nohup python -m src.hardware_cost_model.eval_realtime_sweep     > realtime_sweep.out 2>&1 &


echo "🎯 Sweep complete."
echo "----------------------------------------------------------"
echo "📊 Check logs in ./logs/ and GPU usage with nvidia-smi"
echo "----------------------------------------------------------"
