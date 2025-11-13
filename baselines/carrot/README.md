# CARROT Baselines

CARROT (Cost-Aware Router with Regressor-based Optimization Techniques) predicts quality scores and token counts for different LLMs, enabling cost-aware routing.

## Files

- **`carrot.py`** - All CARROT classes and utilities
  - Data loading and alignment
  - CarrotKNNBaseline and CarrotLinearBaseline
  - CarrotRouter interface
  - Routing utilities

- **`run.py`** - Combined training and evaluation script

## Quick Start

### Training

```bash
# Train both KNN and Linear models
python run.py --mode train \
  --train_dir ../../data_quality/train \
  --output_dir ../../checkpoints/carrot
```

### Evaluation

```bash
# Evaluate trained models
python run.py --mode eval \
  --eval_dir ../../data_quality/eval \
  --output_dir ../../checkpoints/carrot \
  --results_dir ../../results/carrot
```

### Train and Evaluate

```bash
# Do both in one command
python run.py --mode both \
  --train_dir ../../data_quality/train \
  --eval_dir ../../data_quality/eval \
  --output_dir ../../checkpoints/carrot \
  --results_dir ../../results/carrot
```

## Using CARROT Router

```python
from carrot import load_carrot_router

# Load trained router
router = load_carrot_router(
    model_dir='../../checkpoints/carrot',
    model_type='linear'  # or 'knn'
)

# Get predictions for a specific LLM
query = "What is the capital of France?"
embedding = router.encode(query)
quality = router.get_quality(embedding, "Mistral-7B-Instruct-v0.3")
cost = router.get_cost(embedding, "Mistral-7B-Instruct-v0.3")

print(f"Quality: {quality:.4f}, Cost: {cost:.1f}")

# Or use the convenience method
result = router.predict_from_text(query, "Mistral-7B-Instruct-v0.3")
```

## Data Format

CSV files with columns:
- `prompt`: Input query text
- `judge_score`: Quality score (0.0-1.0) from LLM-as-a-Judge
- `output_tokens`: Number of output tokens

Training data: `data_quality/train/*_scored.csv`
Evaluation data: `data_quality/eval/*_eval_scored.csv`

## API Reference

### CarrotRouter

Main interface for quality and cost prediction.

**Methods:**
- `encode(query)` - Convert text to embeddings
- `get_quality(embedding, llm_name)` - Get quality score (0.0-1.0)
- `get_cost(embedding, llm_name)` - Get token count
- `get_quality_all(embedding)` - Get quality for all LLMs (dict)
- `get_cost_all(embedding)` - Get cost for all LLMs (dict)
- `predict_from_text(query, llm_name)` - Encode + predict

**Properties:**
- `available_models` - List of available LLM names

### CarrotKNNBaseline

K-Nearest Neighbors regression (k=256, cosine similarity)

**Methods:**
- `fit(embedding_train, quality_train, token_count_train, save_dir)`
- `predict(embedding_test)` - Returns (quality_pred, token_count_pred)
- `save(save_dir)`
- `load(load_dir)`

### CarrotLinearBaseline

Linear regression with intercept

**Methods:**
- `fit(embedding_train, quality_train, token_count_train, save_dir)`
- `predict(embedding_test)` - Returns (quality_pred, token_count_pred)
- `save(save_dir)`
- `load(load_dir)`

## Command Line Options

```bash
python run.py --help
```

**Required:**
- `--mode {train,eval,both}` - Operation mode

**Optional:**
- `--train_dir` - Training data directory (default: ../../data_quality/train)
- `--eval_dir` - Evaluation data directory (default: ../../data_quality/eval)
- `--output_dir` - Model save/load directory (default: ../../checkpoints/carrot)
- `--results_dir` - Results directory (default: ../../results/carrot)
- `--encoder_model` - Sentence transformer model (default: all-MiniLM-L6-v2)
- `--n_neighbors` - KNN neighbors (default: 256)
- `--seed` - Random seed (default: 42)

## Output

### Training
- `checkpoints/carrot/carrot_knn/` - KNN models
- `checkpoints/carrot/carrot_linear/` - Linear models
- `checkpoints/carrot/metadata.json` - Training metadata

### Evaluation
- `results/carrot/cost_quality_carrot-knn.png` - KNN Pareto frontier
- `results/carrot/cost_quality_carrot-linear.png` - Linear Pareto frontier
- `results/carrot/evaluation_results.json` - Detailed metrics (MSE, MAE, R²)

## Installation

```bash
source .venv/bin/activate
pip install sentence-transformers scikit-learn matplotlib joblib
```

## Notes

- Training and eval sets can have different models (automatically filtered)
- Data is aligned on common prompts across all models
- NaN values and duplicate prompts are automatically removed
- Default encoder: `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings)