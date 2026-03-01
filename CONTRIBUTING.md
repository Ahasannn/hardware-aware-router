# Contributing to HW-Router

Thank you for your interest in contributing to HW-Router.

## Adding a New Router

All routers share the `BaseRouter` interface. To add your own:

1. Subclass `BaseRouter` and implement `compute()`:

```python
from hw_router import BaseRouter

class MyRouter(BaseRouter):
    def compute(self, model_name: str, prompt: str):
        """
        Args:
            model_name: HuggingFace model name (e.g., "Qwen2.5-14B-Instruct")
            prompt: The user's input text

        Returns:
            quality: float in [0, 1], higher is better
            cost: float, lower is better (static price, latency, or custom metric)
        """
        quality = your_quality_function(model_name, prompt)
        cost = your_cost_function(model_name, prompt)
        return quality, cost
```

2. Register it in `hw_router/routers.py` and export from `hw_router/__init__.py`.

3. Add it to the evaluation pipeline in `pipeline/evaluation/eval_lambda_sweep.py`.

See `examples/add_custom_router.py` for a complete working example.

## Adding a New GPU Configuration

Create a YAML file in `configs/`:

```yaml
# configs/my_gpu_setup.yaml
0:  # GPU ID
  - model_name: /path/to/model-a
    port: 8010
  - model_name: /path/to/model-b
    port: 8011
1:  # GPU ID
  - model_name: /path/to/model-c
    port: 8012
```

Then use it with pipeline scripts:

```bash
python pipeline/data_collection/build_hardware_cost_dataset.py \
    --config configs/my_gpu_setup.yaml
```

`configs/test_small_models.yaml` is a minimal 2-model config useful for testing the pipeline without H100s (uses small public models).

## Development Setup

```bash
git clone https://github.com/Ahasannn/hardware-aware-router.git
cd hardware-aware-router
pip install -e ".[dev]"
```

## Running Tests

```bash
# Unit tests only (no GPU/vLLM needed)
pytest tests/ -m "not integration" -v

# Or use make
make test
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting:

```bash
ruff check hw_router/ baselines/
# Or: make lint
```

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Run linter: `make lint`
6. Submit a pull request
