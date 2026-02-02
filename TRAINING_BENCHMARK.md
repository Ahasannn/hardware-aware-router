# Hardware Cost Model Training Benchmark

## System Configuration

**Hardware:**
- **CPU:** AMD Ryzen Threadripper PRO 3955WX 16-Cores
- **Cores:** 16 physical cores (32 logical CPUs with SMT)
- **Base/Boost Clock:** 2.2 GHz / 3.9 GHz
- **Memory:** 125 GB RAM
- **Platform:** Linux (Ubuntu 22.04)

**Software:**
- **Python:** 3.10
- **PyTorch:** (version will be displayed during training)
- **Device:** CPU (training does not require GPU)

## Training Commands

### Default (Auto-detect GPU/CPU):
```bash
python -m src.hardware_cost_model.train_hardware_cost_model
```

### Force CPU Training (for benchmarking):
```bash
# Method 1: Using command-line flag
python -m src.hardware_cost_model.train_hardware_cost_model --cpu

# Method 2: Using environment variable
CUDA_VISIBLE_DEVICES="" python -m src.hardware_cost_model.train_hardware_cost_model
```

**Note:** For benchmarking CPU performance to share with reviewers, use one of the CPU training methods above.

## Model Details

- **Model Type:** Neural Network (HardwareCostNet)
- **Task:** Predicts TTFT (Time To First Token) and TPOT (Time Per Output Token)
- **Training Data:** H100 full sweep dataset
- **Training:** 50 epochs with AdamW optimizer
- **Output:** Model checkpoint and preprocessor saved to `checkpoints/hardware_cost_model/`

## Why CPU Training is Sufficient

1. **Small Model:** The hardware cost predictor is a lightweight neural network designed for fast inference
2. **Small Dataset:** Training data fits comfortably in memory
3. **Fast Convergence:** Only 50 epochs needed for good performance
4. **No Complex Operations:** Simple feedforward network without convolutions or attention mechanisms

## Expected Training Metrics

Training metrics will be reported automatically when you run the script. The updated script now tracks:

**Dataset Information:**
- Total number of samples
- Training/validation split with percentages
- Number of input features
- Total iterations (batches × epochs)

**Timing Information:**
- Data loading time
- Preprocessing time
- Training time (total, per epoch, and per batch)
- Model saving time
- Total wall clock time
- Training throughput (samples/second)

**Example Output:**
```
TRAINING SUMMARY
================================================================================
Device:                cpu
Total dataset size:    XXX,XXX samples
Training samples:      XXX,XXX samples (80.0%)
Validation samples:    XX,XXX samples (20.0%)
Input features:        XX
Output targets:        2 (TTFT + TPOT)
Training epochs:       50
Total iterations:      X,XXX (XXX batches × 50 epochs)
--------------------------------------------------------------------------------
Data loading time:         X.XXs  ( X.X%)
Preprocessing time:        X.XXs  ( X.X%)
Training time:            XX.XXs  (XX.X%)
   • Per epoch:            X.XXs
   • Per batch:           XX.XXms
Model saving time:         X.XXs  ( X.X%)
--------------------------------------------------------------------------------
TOTAL TIME:               XX.XXs  (X.XX minutes)
Throughput:           XX,XXX samples/second
================================================================================
```

**Note:** The model is intentionally designed to be lightweight and train quickly on CPU, making it accessible for researchers and practitioners without GPU access.
