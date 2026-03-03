# Hardware Cost Model: Training Reference

The hardware cost model (HardwareCostNet) is a lightweight MLP that predicts TTFT and TPOT from real-time hardware state. It trains in under a minute on CPU — no GPU required.

## Training Command

```bash
# Auto-detects GPU if available, otherwise uses CPU
python -m pipeline.training.train_cost_model

# Force CPU training
python -m pipeline.training.train_cost_model --cpu

# Custom data path and output directory
python -m pipeline.training.train_cost_model \
    --data data/h100_full_sweep.csv \
    --output-dir checkpoints/hardware_cost_model \
    --epochs 50
```

**Output:** `checkpoints/hardware_cost_model/model.pt` and `checkpoints/hardware_cost_model/preproc.joblib`

## Model Details

| Property | Value |
|----------|-------|
| Architecture | 3-layer MLP with GELU activations |
| Input features | 8 (p_tokens, running/waiting requests, KV-cache usage, TTFT avg, ITL avg, model_id, gpu_id) |
| Output targets | 2 (log-scale TTFT, log-scale TPOT) |
| Training epochs | 50 (AdamW optimizer) |
| Training time | ~20 seconds on CPU |
| Training data | `data/h100_full_sweep.csv` (~8,000 samples) |

## Why CPU Training Is Sufficient

The model is intentionally small: no convolutions, no attention, no large matrix multiplications. The training dataset fits entirely in RAM. GPU acceleration provides no meaningful speedup at this scale.

## Training Output

The script prints a summary when training completes:

```
TRAINING SUMMARY
================================================================================
Device:                cpu
Total dataset size:    8,247 samples
Training samples:      6,597 samples (80.0%)
Validation samples:    1,650 samples (20.0%)
Input features:        8
Output targets:        2 (TTFT + TPOT)
Training epochs:       50
--------------------------------------------------------------------------------
Data loading time:         0.31s
Preprocessing time:        0.18s
Training time:            18.42s
   • Per epoch:            0.37s
Model saving time:         0.04s
--------------------------------------------------------------------------------
TOTAL TIME:               18.95s
Throughput:           17,400 samples/second
================================================================================
```

Validation RMSE for TTFT and TPOT (in log-space) is printed per epoch. Final values should be below 0.15 on the included H100 dataset.
