# Pipeline: Reproducing HW-Router Results

This directory contains all scripts needed to reproduce the HW-Router paper results, organized in execution order.

## Prerequisites

- **Hardware**: 2x NVIDIA H100 GPUs (or equivalent)
- **Software**: vLLM serving framework, Python 3.10+
- **Dependencies**: `pip install -e .` from the repository root

## Pipeline Steps

### Step 1: Data Preparation (`data_preparation/`)

Prepare the combined prompt dataset from MixInstruct and LongBench.

```bash
# Download and process MixInstruct prompts
python pipeline/data_preparation/load_mixinstruct.py

# Download and process LongBench prompts
python pipeline/data_preparation/load_longbench.py

# Combine into train/eval splits (6,740 train + 1,685 eval)
python pipeline/data_preparation/combine_datasets.py

# Save prompt embeddings for CARROT router
python pipeline/data_preparation/save_prompt_embeddings.py \
    --input data/prompts/mixed_prompts_train.parquet \
    --output data/prompts/mixed_prompts_train_with_prompt_embeddings.parquet

# Build UMR training CSV (optional, for UMR baseline)
python pipeline/data_preparation/build_umr_training_csv.py
```

**Output**: `data/prompts/mixed_prompts_{train,eval}.parquet`

### Step 2: Data Collection (`data_collection/`)

Collect hardware-aware latency data from live vLLM instances. Requires running vLLM servers (see `infrastructure/vllm/`).

```bash
# Collect training data with various arrival patterns
python pipeline/data_collection/build_hardware_cost_dataset.py \
    --config configs/gpu_model_map_h100.yaml \
    --output data/h100_full_sweep.csv \
    --pattern poisson --rate 5.0

# Collect evaluation data
python pipeline/data_collection/build_eval_dataset.py \
    --config configs/gpu_model_map_h100.yaml \
    --output data/evaluation_dataset.csv

# Compute normalization constants (LAT_P95_LOG, STATIC_COST_P95)
python pipeline/data_collection/compute_normalization.py
```

**Output**: `data/h100_full_sweep.csv`, `data/evaluation_dataset.csv`

### Step 3: Training (`training/`)

Train the hardware cost model (lightweight MLP).

```bash
python -m pipeline.training.train_cost_model
# Or with CPU-only: python -m pipeline.training.train_cost_model --cpu
```

**Output**: `checkpoints/hardware_cost_model/model.pt`, `checkpoints/hardware_cost_model/preproc.joblib`

### Step 4: Evaluation Processing (`eval_processing/`)

Process the evaluation dataset with router predictions.

```bash
# Add CARROT quality/length predictions + HW cost predictions
python pipeline/eval_processing/process_eval_dataset.py \
    --input data/evaluation_dataset.csv \
    --output data/evaluation_dataset_processed_full.csv

# Add UMR quality scores (optional, for UMR baseline)
python pipeline/eval_processing/update_eval_with_umr.py

# Add IRT quality scores
python pipeline/eval_processing/update_eval_with_irt.py
```

**Output**: `data/evaluation_dataset_processed_full.csv`

### Step 5: Evaluation (`evaluation/`)

Run offline and online evaluations.

```bash
# Offline: Lambda sweep (generates data for Figure 4a, 4b)
python pipeline/evaluation/eval_lambda_sweep.py \
    --eval_csv data/evaluation_dataset_processed_full.csv

# Online: Live vLLM evaluation (requires running vLLM servers)
python pipeline/evaluation/eval_runtime_router.py \
    --config configs/gpu_model_map_h100.yaml \
    --prompt_path data/prompts/mixed_prompts_eval.parquet \
    --eval_csv data/evaluation_dataset_processed_full.csv \
    --router hw

# Online: Sweep over arrival rates
python pipeline/evaluation/eval_realtime_sweep.py \
    --arrival_rates "15,18,21" \
    --pattern_type sustained

# Full pipeline evaluation (all routers)
python pipeline/evaluation/eval_pipeline.py \
    --config configs/gpu_model_map_h100.yaml \
    --prompt_path data/prompts/mixed_prompts_eval.parquet
```

## Expected Results

After running the full pipeline, you should observe:
- **HW-Router vs CARROT**: 3.4-3.9x lower average latency
- **SLO attainment**: 46-48 percentage point improvement
- **Routing overhead**: < 1ms per request
