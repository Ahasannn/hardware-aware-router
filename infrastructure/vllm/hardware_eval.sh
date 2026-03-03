python -m src.hardware_cost_model.eval_pipeline \
    --config configs/gpu_model_map_h100.yaml \
    --prompt_path data/prompts/mixed_prompts_eval.parquet \
    --output data/eval_results/carrot_hw.csv \
    --router carrot \
    --use_hw_cost \
    --num_prompts 200


python -m src.hardware_cost_model.eval_pipeline \
    --config configs/gpu_model_map_qwen.yaml \
    --prompt_path data/prompts/mixed_prompts_eval.parquet \
    --output data/eval_results/carrot_static.csv \
    --router carrot \
    --num_prompts 200


nohup python -m src.hardware_cost_model.build_router_evaluation_dataset \
    --config configs/gpu_model_map_h100.yaml \
    --output data/evaluation_dataset_full.csv \
    --num_prompts 2000 \
    --concurrency 120 \
    --pattern poisson \
    --rate 18 \
    --prompt_path data/prompts/mixed_prompts_eval.parquet \
    > logs/evaluation_dataset_full.log 2>&1 &

