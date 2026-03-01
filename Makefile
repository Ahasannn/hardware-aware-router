.PHONY: setup setup-all data train reproduce test lint clean help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Install core package (no vLLM dependency)
	pip install -e .

setup-all:  ## Install package with all optional dependencies
	pip install -e ".[all]"

data:  ## Generate prompt datasets from public sources (MixInstruct + LongBench)
	python pipeline/data_preparation/load_mixinstruct.py
	python pipeline/data_preparation/load_longbench.py
	python pipeline/data_preparation/combine_datasets.py

train:  ## Train the cost predictor (~20 seconds on CPU)
	python -m pipeline.training.train_cost_model

reproduce:  ## Reproduce paper figures (no GPU needed)
	python scripts/reproduce_figures.py

test:  ## Run unit tests
	pytest tests/ -m "not integration" -v

lint:  ## Run ruff linter
	ruff check hw_router/ baselines/ pipeline/

clean:  ## Remove generated artifacts
	rm -rf __pycache__ .pytest_cache *.egg-info build/ dist/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
