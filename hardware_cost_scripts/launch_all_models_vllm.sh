#!/bin/bash
# ==========================================================
# Multi-Model vLLM Deployment Script
# For 2 × 24 GB GPUs, each hosting 2 models
# ==========================================================

# Kill any running vLLM servers to free GPU memory
echo "🔪 Killing existing vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server"
sleep 3

# Make sure log directory exists
mkdir -p logs
echo "🚀 Starting new vLLM servers ..."

# ---------------------- GPU 0 ----------------------
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen1.5-0.5B \
  --port 8005 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.4 \
  > logs/qwen0_0.5b.log 2>&1 &

sleep 20

CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --port 8006 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.3 \
  > logs/qwen0_1.5b.log 2>&1 &

sleep 20

# ---------------------- GPU 1 ----------------------
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4 \
  --port 8007 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.3 \
  > logs/qwen1_0.5B-Chat-GPTQ-Int4.log 2>&1 &

sleep 20

CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct-AWQ \
  --port 8008 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.45 \
  > logs/qwen1_1.7b.log 2>&1 &

echo "----------------------------------------------------------"
echo "✅ All vLLM servers launched in background."
echo "Logs: ./logs/"
echo "Check running processes: ps aux | grep vllm"
echo "Monitor GPU usage: watch -n 1 nvidia-smi"
echo "----------------------------------------------------------"
