# Data Directory

This directory contains the datasets used for training and evaluation. Most data files are gitignored due to size.

## Obtaining the Full Data

### Option 1: Generate from scratch

Follow the pipeline steps in [pipeline/README.md](../../pipeline/README.md) to generate all datasets from scratch.

### Option 2: Download pre-generated data

Contact the authors for pre-generated datasets.

## Expected Directory Layout

```
data/
├── h100_full_sweep.csv                          # Training data (hardware metrics + latency)
├── evaluation_dataset.csv                       # Raw evaluation data
├── evaluation_dataset_processed_full.csv        # Processed eval data with router predictions
├── UMR_router_training_data.csv                 # UMR baseline training data
├── prompts/
│   ├── mixed_prompts_train.parquet              # 6,740 training prompts
│   ├── mixed_prompts_eval.parquet               # 1,685 evaluation prompts
│   ├── mixed_prompts_train_with_prompt_embeddings.parquet
│   └── mixed_prompts_eval_with_prompt_embeddings.parquet
├── data_quality/                                # Per-model quality scores
│   ├── qwen14b_scored.csv
│   ├── phi3_scored.csv
│   ├── llama3_scored.csv
│   ├── qwen3b_scored.csv
│   └── mistral7b_scored.csv
├── figures/                                     # Generated figures (PDF)
├── cost_predictor_plots/                        # Cost model evaluation plots
└── sample/
    └── README.md                                # This file
```

## Key Columns

### h100_full_sweep.csv (Training Data)
- `p_tokens`: Prompt token count
- `model_id`: Local model name
- `gpu_id`: GPU ID (0 or 1)
- `running_req_count`: Number of running requests at dispatch time
- `waiting_req_count`: Number of waiting requests
- `kv_cache_usage_perc`: KV-cache utilization percentage
- `ttft_avg`, `itl_avg`, `e2e_avg`: Rolling average latency metrics
- `ttft_s`: Time to first token (seconds)
- `tpot_s_per_token`: Time per output token (seconds)
- `latency_s`: End-to-end latency (seconds)
- `d_tokens`: Number of decoded (output) tokens

### evaluation_dataset_processed_full.csv (Evaluation Data)
All columns from above, plus:
- `carrot_predicted_quality`: CARROT quality score
- `carrot_predicted_length`: CARROT predicted output length
- `irt_quality_score`: IRT quality score
- `umr_quality_score`: UMR quality score
