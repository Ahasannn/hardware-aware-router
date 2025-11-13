import torch, numpy as np, joblib

class HardwareCostPredictor:
    """Wrapper for your trained HardwareCostNet model (TTFT, TPOT prediction)."""
    def __init__(self, model_path: str, kind: str = "torch"):
        self.kind = kind
        if kind == "torch":
            self.model = torch.load(model_path, map_location="cpu")
            self.model.eval()
        elif kind == "joblib":
            self.model = joblib.load(model_path)
        else:
            raise ValueError(f"Unsupported model type: {kind}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """features: (N, F). Returns (N, 2): [ttft_hat, tpot_hat]."""
        if self.kind == "torch":
            with torch.no_grad():
                y = self.model(torch.tensor(features, dtype=torch.float32))
            return y.cpu().numpy()
        elif self.kind == "joblib":
            return self.model.predict(features)
