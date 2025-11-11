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




# Low load, Poisson pattern
python build_hardware_cost_dataset.py --config gpu_map.yaml --pattern poisson --rate 2 --concurrency 5

# Medium load
python build_hardware_cost_dataset.py --config gpu_map.yaml --pattern poisson --rate 5 --concurrency 10

# High load
python build_hardware_cost_dataset.py --config gpu_map.yaml --pattern poisson --rate 12 --concurrency 20

# Bursty
python build_hardware_cost_dataset.py --config gpu_map.yaml --pattern microburst --rate 5 --concurrency 20

# Sustained overload
python build_hardware_cost_dataset.py --config gpu_map.yaml --pattern sustained --rate 10 --concurrency 20
