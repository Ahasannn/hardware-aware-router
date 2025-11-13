# routers_baseline.py
import random

# Dummy table: model -> parameter count (for quality score)
MODEL_PARAMS = {
    "Phi-3-mini": 4e9,
    "Qwen-3B": 3e9,
    "Mistral-7B": 7e9,
    "Llama-8B": 8e9,
    "Qwen-14B": 14e9,
}

class BaselineRouter:
    """Pure quality-based (param-size). Cost = 0."""
    def __init__(self, model_names):
        self.models = model_names

    def score(self, model_name):
        return MODEL_PARAMS.get(model_name, 1e9)

    def route(self, prompt):
        scores = {m: self.score(m) for m in self.models}
        return max(scores, key=scores.get)  # highest quality
        

class BaselineRouterWithOurCost:
    """
    quality = model params
    cost = predicted TTFT + TPOT
    score = quality - lambda * cost
    """
    def __init__(self, model_names, cost_predictor, lam=1.0):
        self.models = model_names
        self.cost_predictor = cost_predictor
        self.lam = lam

    def quality(self, m):
        from .routers_baseline import MODEL_PARAMS
        return MODEL_PARAMS.get(m, 1e9)

    def route(self, prompt, per_model_features):
        """
        per_model_features: dict {model_name: feature_dict_for_that_model}
        """
        scores = {}
        for m in self.models:
            q = self.quality(m)
            ttft, tpot = self.cost_predictor(m, per_model_features[m])
            cost = ttft + tpot
            scores[m] = q - self.lam * cost

        return max(scores, key=scores.get)

