# routers.py

from abc import ABC, abstractmethod
import random

# Simple quality proxy for baseline router (extend as needed)
MODEL_QUALITY = {
    "Qwen/Qwen2.5-14B-Instruct" : .14,
    "microsoft/Phi-3-mini-128k-instruct" : .3,
    "meta-llama/Llama-3.1-8B-Instruct" : .8,
    "Qwen/Qwen2.5-3B-Instruct" : .3,
    "mistralai/Mistral-7B-Instruct-v0.3" : .7
}

MODEL_PRICES = {
    "Qwen/Qwen2.5-14B-Instruct" : 0.22 / 1000000,
    "microsoft/Phi-3-mini-128k-instruct" : .10 / 1000000,
    "meta-llama/Llama-3.1-8B-Instruct" : 0.03 / 1000000,
    "Qwen/Qwen2.5-3B-Instruct" : 0.05 / 1000000,
    "mistralai/Mistral-7B-Instruct-v0.3" : 0.20 / 1000000
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
    def __init__(self, carrot_model):
        """
        carrot_model: object returned by load_carrot_router(...)
        """
        self.carrot = carrot_model

    def compute(self, model_name, prompt):
        emb = self.carrot.encode(prompt)

        # CARROT quality
        q = self.carrot.get_quality(emb, model_name)

        # CARROT cost (static)
        static_cost = self.carrot.get_cost(emb, model_name)
        static_cost = static_cost * MODEL_PRICES.get(model_name, 1e-7)

        return q, static_cost

    def length_predictor(self,model_name,prompt):
        emb = self.carrot.encode(prompt)
        length = self.carrot.get_cost(emb, model_name)
        
        return length
