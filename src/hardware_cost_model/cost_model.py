# cost_model.py

import numpy as np
import torch
import pandas as pd
from joblib import load
from .model_utils import HardwareCostNet


class HardwareCostPredictor:
    """
    Wrapper around trained hardware cost model.
    Training features (per row):

        Numerical:
            - p_tokens
            - running_req_count
            - waiting_req_count
            - kv_cache_usage_perc
            - ttft_avg
            - itl_avg

        Categorical:
            - model_gpu = model_id + "_" + gpu_id

    During evaluation you must provide:
        feat = {
            "p_tokens": int,
            "running_req_count": int,
            "waiting_req_count": int,
            "kv_cache_usage_perc": float,
            "ttft_avg": float,
            "itl_avg": float,
            "model_id": str,
            "gpu_id": str,   # e.g. "0", "1"
        }
    """

    def __init__(self, model_path: str, preproc_path: str):
        # Load preprocessor (ColumnTransformer on the above features)
        self.preproc = load(preproc_path)

        # Build model with correct input dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = len(self.preproc.get_feature_names_out())

        self.model = HardwareCostNet(input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def __call__(self, model_name: str, feat: dict):
        """
        Predict TTFT and TPOT (original scale) for a single (model, feature) pair.
        """
        df = self._prepare_df(model_name, feat)
        X = self.preproc.transform(df)

        with torch.no_grad():
            Xv = torch.tensor(X, dtype=torch.float32).to(self.device)
            ttft_log, tpot_log = self.model(Xv)

        # Back-transform from log to original scale
        ttft = float(np.exp(ttft_log.cpu().numpy().squeeze()))
        tpot = float(np.exp(tpot_log.cpu().numpy().squeeze()))

        return ttft, tpot

    def _prepare_df(self, model_name: str, feat: dict) -> pd.DataFrame:
        """
        Construct single-row DataFrame in EXACT training format.
        """
        data = {
            "p_tokens": feat["p_tokens"],
            "running_req_count": feat["running_req_count"],
            "waiting_req_count": feat["waiting_req_count"],
            "kv_cache_usage_perc": feat["kv_cache_usage_perc"],
            "ttft_avg": feat["ttft_avg"],
            "itl_avg": feat["itl_avg"],
            "model_gpu": f"{model_name}_{feat['gpu_id']}",
        }
        return pd.DataFrame([data])
