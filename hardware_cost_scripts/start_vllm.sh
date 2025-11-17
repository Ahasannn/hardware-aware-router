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


CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --port 8015 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.4 \
  > logs/qwen0_2.5-14b-instruct.log 2>&1 &

  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model microsoft/Phi-3-mini-128k-instruct \
  --port 8016 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.2 \
  > logs/gpu0_Phi-3-mini-128k-instruct.log 2>&1 &


  CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model microsoft/Phi-3-mini-128k-instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.6 \
  --dtype float16 \
  --enforce-eager \
  --port 8016 --host 0.0.0.0

CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model microsoft/Phi-3-mini-128k-instruct \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.3 \
  --port 8016 \
  --host 0.0.0.0 \
  > logs/gpu0_phi3_mini.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model microsoft/Phi-3-mini-128k-instruct \
  --max-model-len 21030 \
  --gpu-memory-utilization 0.2 \
  --port 8016 \
  --host 0.0.0.0 \
  > logs/gpu0_phi3_mini.log 2>&1 &

  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.6 \
  --port 8016 \
  --host 0.0.0.0 \
  > logs/gpu0_qwen2.5_14b.log 2>&1 &

  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model microsoft/Phi-3-mini-128k-instruct \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.3 \
  --port 8015 \
  --host 0.0.0.0 \
  > logs/gpu0_phi3_mini.log 2>&1 &

  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.2 \
  --port 8017 \
  --host 0.0.0.0 \
  > logs/gpu1_qwen_2.5-3b.log 2>&1 &

  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.3 \
  --port 8018 \
  --host 0.0.0.0 \
  > logs/gpu1_mistral_7b_instruct.log 2>&1 &

  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.4 \
  --port 8018 \
  --host 0.0.0.0 \
  > logs/gpu1_llama-3.1-8b.log 2>&1 &

  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/llama3-8b \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.4 \
  --port 8010 \
  --host 0.0.0.0 \
  > logs/gpu0-llama3-8b.log 2>&1 &

  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen3b \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.2 \
  --port 8011 \
  --host 0.0.0.0 \
  > logs/gpu0-qwen3b.log 2>&1 &

  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/mistral7b \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.3 \
  --port 8012 \
  --host 0.0.0.0 \
  > logs/gpu0_mistral7b.log 2>&1 &



  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen14b \
  --max-model-len 21000 \
  --gpu-memory-utilization 0.55 \
  --port 8010 \
  --host 0.0.0.0 \
  > logs/gpu0_qwen2.5_14b.log 2>&1 &

  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/phi3-mini \
  --max-model-len 21000 \
  --gpu-memory-utilization 0.25 \
  --port 8011 \
  --host 0.0.0.0 \
  > logs/gpu0_phi3_mini.log 2>&1 &