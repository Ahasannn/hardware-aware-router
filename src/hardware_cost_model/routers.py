# routers.py

from abc import ABC, abstractmethod
import random

# Simple quality proxy for baseline router (extend as needed)
MODEL_QUALITY = {
    "Qwen/Qwen1.5-0.5B": 0.5,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4": 0.45,
    "Qwen/Qwen2.5-0.5B-Instruct-AWQ": 0.5,
}

# ============================================================
# Base Router Interface
# ============================================================
class BaseRouter(ABC):
    @abstractmethod
    def compute(self, model_name: str, prompt: str):
        """
        Return:
            quality_score (float)
            cost_score (float)  # static or router-defined cost
        """
        pass


# ============================================================
# Baseline Router (quality-only OR static cost = 0)
# ============================================================
class BaselineRouter(BaseRouter):
    def compute(self, model_name, prompt):
        quality = MODEL_QUALITY.get(model_name, 1.0)
        cost = 0.0
        return quality, cost


# ============================================================
# Random Router (for sanity)
# ============================================================
class RandomRouter(BaseRouter):
    def compute(self, model_name, prompt):
        quality = random.random()
        cost = random.random()
        return quality, cost


# ============================================================
# Round Robin with dummy scoring
# ============================================================
class RoundRobinRouter(BaseRouter):
    def __init__(self):
        self.counter = 0  

    def compute(self, model_name, prompt):
        quality = MODEL_QUALITY.get(model_name, 1.0)
        cost = 0.0
        return quality, cost


# ============================================================
# CARROT Router (static cost + CARROT quality)
# ============================================================
class CarrotRouter(BaseRouter):
    def __init__(self, carrot_model, model_prices):
        """
        carrot_model: object returned by load_carrot_router(...)
        model_prices: dict {model_name: price-per-token}
        """
        self.carrot = carrot_model
        self.prices = model_prices

    def compute(self, model_name, prompt):
        emb = self.carrot.encode(prompt)

        # CARROT quality
        q = self.carrot.get_quality(emb, model_name)

        # CARROT cost (static)
        static_cost = self.carrot.get_cost(emb, model_name)
        static_cost = static_cost * self.prices.get(model_name, 1e-7)

        return q, static_cost
