# cost_model.py

import numpy as np
import torch
import pandas as pd
from joblib import load
from .model_utils import HardwareCostNet


class HardwareCostPredictor:
    """
    Hardware cost model predictor.

    Training features:
        Numerical:
            - p_tokens
            - running_req_count
            - waiting_req_count
            - kv_cache_usage_perc
            - ttft_avg
            - itl_avg

        Categorical:
            - model_gpu = "<model_id_int>_<gpu_id_str>"

    During prediction, always call:
        predictor(model_id_int, feat_dict)

    Where feat_dict contains:
        {
            "p_tokens": int,
            "running_req_count": int,
            "waiting_req_count": int,
            "kv_cache_usage_perc": float,
            "ttft_avg": float,
            "itl_avg": float,
            "model_id": int,     # must be int 0..4
            "gpu_id": str,       # "0", "1", ...
        }
    """

    def __init__(self, model_path: str, preproc_path: str):
        # Load the ColumnTransformer used during training
        self.preproc = load(preproc_path)

        # Build neural network with correct input dimension
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = len(self.preproc.get_feature_names_out())

        self.model = HardwareCostNet(input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    # -------------------------------------------------------
    # NOTE: model_id here is ALWAYS an integer (0..4)
    # -------------------------------------------------------
    def __call__(self, model_id: int, feat: dict):
        """
        Predict TTFT and TPOT (original scale).
        """
        df = self._prepare_df(model_id, feat)
        X = self.preproc.transform(df)

        with torch.no_grad():
            Xv = torch.tensor(X, dtype=torch.float32).to(self.device)
            ttft_log, tpot_log = self.model(Xv)

        # Back-transform from log-space
        ttft = float(np.exp(ttft_log.cpu().numpy().squeeze()))
        tpot = float(np.exp(tpot_log.cpu().numpy().squeeze()))

        return ttft, tpot

    # -------------------------------------------------------
    # Construct DataFrame exactly matching training schema
    # -------------------------------------------------------
    def _prepare_df(self, model_id: int, feat: dict) -> pd.DataFrame:
        """
        Build single-row DataFrame with exact training columns.
        """
        data = {
            "p_tokens": feat["p_tokens"],
            "running_req_count": feat["running_req_count"],
            "waiting_req_count": feat["waiting_req_count"],
            "kv_cache_usage_perc": feat["kv_cache_usage_perc"],
            "ttft_avg": feat["ttft_avg"],
            "itl_avg": feat["itl_avg"],
            "model_gpu": f"{model_id}_{feat['gpu_id']}",
        }
        return pd.DataFrame([data])
