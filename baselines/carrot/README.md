# CARROT Baselines

CARROT (Cost-Aware Router with Regressor-based Optimization Techniques) predicts quality scores and token counts for different LLMs, enabling cost-aware routing.

## Files

- **`carrot.py`** — CarrotKNNBaseline, CarrotLinearBaseline, CarrotRouter, and utilities
- **`checkpoints/carrot/`** — Pre-trained KNN and linear model weights

## Using CARROT Router

```python
from baselines.carrot.carrot import load_carrot_router

router = load_carrot_router(
    model_dir='checkpoints/carrot',
    model_type='linear'  # or 'knn'
)

quality = router.get_quality(router.encode("Your query"), "Mistral-7B-Instruct-v0.3")
cost    = router.get_cost(router.encode("Your query"),    "Mistral-7B-Instruct-v0.3")
```

## Data Format

CSV files with columns:
- `prompt` — Input query text
- `judge_score` — Quality score (0.0–1.0) from LLM-as-a-Judge
- `output_tokens` — Number of output tokens

See `data/data_quality/` for the committed judge scores.

## API Reference

### CarrotRouter

| Method | Description |
|--------|-------------|
| `encode(query)` | Convert text to embeddings |
| `get_quality(embedding, llm_name)` | Quality score (0.0–1.0) |
| `get_cost(embedding, llm_name)` | Predicted token count |
| `predict_from_text(query, llm_name)` | Encode + predict in one call |

### Model Classes

- **CarrotKNNBaseline** — k-NN regression (k=256, cosine similarity)
- **CarrotLinearBaseline** — Linear regression with intercept
