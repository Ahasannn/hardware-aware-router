#!/bin/bash
# Collect hardware-aware latency dataset

python -m src.hardware_cost_model.build_hardware_cost_dataset \
  --config configs/gpu_model_map.yaml \  
  --output data/hw_dataset_highload.csv \ 
  --num_prompts 1000 \ 
  --interval 1 \ 
  --concurrency 100