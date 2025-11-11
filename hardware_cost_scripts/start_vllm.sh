#!/bin/bash
# Start vLLM server

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --port 8005 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.3

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model microsoft/Phi-3-mini-128k-instruct \
  --port 8005 \
  --host 0.0.0.0

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen1.5-0.5B \
  --port 8005 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.3

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 8006 \
  --host 0.0.0.0

  CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --port 8007 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.35
