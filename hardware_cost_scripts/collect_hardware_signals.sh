#!/bin/bash
# Collect hardware-aware latency dataset

nohup python -m src.hardware_cost_model.build_hardware_cost_dataset \
  --config configs/gpu_model_map_qwen.yaml \
  --output data/hw_dataset_qween_4_test.csv \
  --num_prompts 60 \
  --interval 0.3 \
  --pattern poisson \
  --rate 12 \
  --concurrency 20 \
  > logs/build_hw_dataset_run_qween_test.log 2>&1 &


