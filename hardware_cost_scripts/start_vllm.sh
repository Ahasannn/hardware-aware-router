#!/bin/bash
# Start vLLM server

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --port 8005 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.3


# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
#   --model Qwen/Qwen1.5-0.5B \
#   --port 8006 \
#   --host 0.0.0.0 \
#   --gpu-memory-utilization 0.3

  