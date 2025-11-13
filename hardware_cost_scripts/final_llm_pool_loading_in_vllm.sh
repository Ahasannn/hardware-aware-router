  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen14b \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.6 \
  --port 8010 \
  --host 0.0.0.0 \
  > logs/gpu0_qwen2.5_14b.log 2>&1 &

  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/phi3-mini \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.3 \
  --port 8011 \
  --host 0.0.0.0 \
  > logs/gpu0_phi3_mini.log 2>&1 &


  
  
  CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/llama3-8b \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.4 \
  --port 8012 \
  --host 0.0.0.0 \
  > logs/gpu1-llama3-8b.log 2>&1 &

  CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen3b \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.2 \
  --port 8013 \
  --host 0.0.0.0 \
  > logs/gpu1-qwen3b.log 2>&1 &

  CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/mistral7b \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.3 \
  --port 8014 \
  --host 0.0.0.0 \
  > logs/gpu1_mistral7b.log 2>&1 &

