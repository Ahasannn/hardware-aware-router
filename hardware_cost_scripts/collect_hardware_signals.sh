#!/bin/bash
# Collect hardware-aware latency dataset

nohup python -m src.hardware_cost_model.build_hardware_cost_dataset \
  --config configs/gpu_model_map_h100.yaml \
  --output data/hw_dataset_one_test.csv \
  --num_prompts 60 \
  --interval 0.3 \
  --pattern poisson \
  --rate 12 \
  --concurrency 20 \
  > logs/hw_dataset_one_test.log 2>&1 &

python -m src.hardware_cost_model.build_hardware_cost_dataset \
  --config configs/gpu_model_map_h100.yaml \
  --output data/hw_dataset_one_test.csv \
  --num_prompts 60 \
  --interval 0.3 \
  --pattern poisson \
  --rate 12 \
  --concurrency 20 
