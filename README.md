# HW-Router: Hardware-Aware Routing for Scalable Multi-LLM Serving

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DAC 2026](https://img.shields.io/badge/DAC-2026-green.svg)](https://dac.com)

> **Accepted at the 63rd Design Automation Conference (DAC), 2026**

## Overview

![HW-Router Overview](assets/overview.png)

HW-Router is a hardware-aware routing framework for multi-LLM serving that dynamically selects the best model for each incoming request based on both predicted response quality and real-time hardware conditions.

Unlike static routing approaches that ignore server load, HW-Router integrates a lightweight neural cost predictor that estimates per-request latency (TTFT and TPOT) from live hardware metrics (queue depths, KV-cache utilization, GPU load). Combined with an IRT-based quality predictor, this enables quality-cost trade-off decisions that respect Service Level Objectives (SLOs).

## Architecture

![HW-Router Methodology](assets/methodology.png)

**Components:**

| Component | Module | Description |
|-----------|--------|-------------|
| Hardware Monitor | `hw_router.hardware_monitor` | Polls vLLM Prometheus metrics in real-time |
| Cost Predictor | `hw_router.cost_predictor` | Lightweight MLP predicting TTFT and TPOT — **plug-in component** |
| Quality Predictor | `hw_router.routers` | Any quality predictor: CARROT, IRT, UMR, or custom |
| Decision Maker | `pipeline/evaluation/` | Scores each model: S = λ·Q − (1−λ)·C, picks argmax |

### Modular Design

The hardware cost predictor is a **plug-in** — it works with any quality predictor by replacing the static price/token cost term with real-time hardware-aware latency predictions:

```
Quality-only router:   S = λ · Q(x)       − (1−λ) · static_price/token
Hardware-aware (+HW):  S = λ · Q(x)       − (1−λ) · C(x, h)   ← same Q, HW cost swapped in
                                                      ↑
                                             MLP predicts TTFT + TPOT
                                             from live hardware state h
```

This means **any quality predictor can be made hardware-aware** simply by pairing it with the cost predictor. In the paper we use IRT as the quality component because it yields the best results, but CARROT and UMR benefit equally from hardware cost awareness:

| Quality Predictor | Without HW Cost | With HW Cost (+) | SLO lift |
|-------------------|-----------------|------------------|----------|
| CARROT | 44.7% SLO, 43.9s | 96.1% SLO, 14.4s | +51pp |
| IRT | 42.2% SLO, 45.3s | 97.9% SLO, 12.9s | +56pp ⭐ |
| UMR | 37.3% SLO, 48.4s | 91.5% SLO, 16.7s | +54pp |

*Evaluated at λ = 0.5. IRT+HW is the configuration reported as "HW-Router" in the paper.*

## Quick Start

### Installation

```bash
git clone https://github.com/Ahasannn/hardware-aware-router.git
cd hardware-aware-router
pip install -e .
```

To install with specific components only:

```bash
pip install -e ".[irt]"       # Core + IRT quality predictor
pip install -e ".[carrot]"    # Core + CARROT baseline
pip install -e ".[serving]"   # Core + vLLM serving stack
pip install -e ".[all]"       # Everything
```

### Using the Routers

All routers share the same interface — `compute(model_name, prompt) -> (quality, cost)`:

```python
from hw_router import BaselineRouter, IRTRouter, CarrotRouter, UMRRouter
from hw_router.constants import DEFAULT_LAMBDA

# Choose a quality predictor
router = BaselineRouter()  # Static quality lookup (no dependencies)
# router = IRTRouter()     # IRT-based quality (requires transformers)
# router = UMRRouter()     # Cluster-based quality (requires sentence-transformers)

# Score each candidate model
prompt = "Explain quantum computing in simple terms."
models = ["Qwen2.5-14B-Instruct", "Llama-3.1-8B-Instruct", "Qwen2.5-3B-Instruct"]

for model in models:
    quality, cost = router.compute(model, prompt)
    score = DEFAULT_LAMBDA * quality - (1 - DEFAULT_LAMBDA) * cost
    print(f"{model}: quality={quality:.3f}, score={score:.4f}")
```

### Reproduce Paper Figures

No GPU needed — runs on any laptop using the included evaluation data:

```bash
make reproduce
```

Or manually:

```bash
python scripts/reproduce_figures.py
```

### Running on Your Own Hardware

Want to run HW-Router with your own GPUs and models? See **[docs/CUSTOM_HARDWARE_GUIDE.md](docs/CUSTOM_HARDWARE_GUIDE.md)** for a step-by-step walkthrough covering LLM pool config, vLLM setup, data collection, training, and evaluation.

### Running the Full Pipeline

> **Prerequisites:** Steps 2 and 5 require live vLLM servers and at least 2× NVIDIA H100 GPUs (or equivalent). Steps 1, 3, and 4 run on CPU only. See [docs/CUSTOM_HARDWARE_GUIDE.md](docs/CUSTOM_HARDWARE_GUIDE.md) if you are adapting this to your own hardware.

See [pipeline/README.md](pipeline/README.md) for the complete reproduction guide. The key steps are:

```bash
# 1. Prepare datasets (downloads from public sources — CPU only)
python pipeline/data_preparation/load_mixinstruct.py
python pipeline/data_preparation/load_longbench.py
python pipeline/data_preparation/combine_datasets.py

# 2. Collect hardware data (requires live vLLM servers + H100 GPUs)
python pipeline/data_collection/build_hardware_cost_dataset.py \
    --config configs/gpu_model_map_h100.yaml

# 3. Train cost model — CPU only, ~20 seconds
python -m pipeline.training.train_cost_model

# 4. Run offline evaluation (lambda sweep — CPU only)
python pipeline/evaluation/eval_lambda_sweep.py \
    --eval_csv data/evaluation_dataset_processed_full_with_umr_irt.csv

# 5. Run online evaluation (requires live vLLM servers + H100 GPUs)
python pipeline/evaluation/eval_runtime_router.py \
    --config configs/gpu_model_map_h100.yaml --router hw
```

## Repository Structure

```
hw-router/
├── hw_router/              # Core library (pip-installable)
│   ├── constants.py        # Prices, normalization constants, quality proxies
│   ├── model_registry.py   # Model name → ID mappings (no hardcoded paths)
│   ├── cost_predictor.py   # Neural cost model (HardwareCostNet + predictor)
│   ├── routers.py          # Quality predictors (IRT, CARROT, UMR, baselines)
│   ├── hardware_monitor.py # Real-time vLLM metrics polling
│   └── load_patterns.py    # Request arrival patterns (Poisson, microburst, sustained)
│
├── baselines/              # Baseline router implementations
│   ├── carrot/             # CARROT router (KNN + Linear)
│   ├── irt/                # IRT/MIRT quality predictor
│   └── umr/                # Unified Model Router (clustering-based)
│
├── pipeline/               # Reproducibility pipeline (see pipeline/README.md)
│   ├── data_preparation/   # Step 1: Prepare datasets
│   ├── data_collection/    # Step 2: Collect hardware data from vLLM
│   ├── training/           # Step 3: Train cost model
│   ├── eval_processing/    # Step 4: Process evaluation datasets
│   └── evaluation/         # Step 5: Run evaluations (offline + online)
│
├── analysis/               # Post-hoc analysis and visualization
│   ├── plots/              # Figure generation scripts
│   └── notebooks/          # Jupyter notebooks for exploration
│
├── examples/               # Usage examples
├── configs/                # GPU-model YAML configuration maps
├── scripts/                # Utilities (scoring, figure reproduction)
├── infrastructure/         # Deployment (SLURM jobs, vLLM launch scripts)
├── data/                   # Datasets (see data/sample/README.md)
├── checkpoints/            # Model weights
├── tests/                  # Test suite
└── docs/                   # Additional documentation
```

## Models

The framework is evaluated with 5 LLMs across 2 NVIDIA H100 GPUs:

| Model | Parameters | GPU |
|-------|-----------|-----|
| Qwen2.5-14B-Instruct | 14B | GPU 0 |
| Phi-3-mini-128k-instruct | 3.8B | GPU 0 |
| Llama-3.1-8B-Instruct | 8B | GPU 1 |
| Qwen2.5-3B-Instruct | 3B | GPU 1 |
| Mistral-7B-Instruct-v0.3 | 7B | GPU 1 |

## Adding Your Own Router

Subclass `BaseRouter` and implement the `compute()` method:

```python
from hw_router import BaseRouter

class MyRouter(BaseRouter):
    def compute(self, model_name: str, prompt: str):
        quality = your_quality_function(model_name, prompt)
        cost = your_cost_function(model_name, prompt)
        return quality, cost
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

## Citation

```bibtex
@inproceedings{kabir2026hwrouter,
  title     = {{HW-Router}: Hardware-Aware Routing for Scalable Multi-{LLM} Serving},
  author    = {Kabir, Ahasan and Xue, Jiaqi and Zheng, Mengxin and Lou, Qian},
  booktitle = {Proceedings of the 63rd Design Automation Conference (DAC)},
  year      = {2026}
}
```

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
