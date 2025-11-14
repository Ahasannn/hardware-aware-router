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
