# cost_model.py
import numpy as np
import torch
from joblib import load
from .model_utils import HardwareCostNet

class HardwareCostPredictor:
    def __init__(self, model_path, preproc_path):
        self.preproc = load(preproc_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = len(self.preproc.get_feature_names_out())
        self.model = HardwareCostNet(input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def __call__(self, model_name, features_dict):
        """
        features_dict = {
            'p_tokens': int,
            'running_req_count': int,
            'waiting_req_count': int,
            'kv_cache_usage_perc': float,
            'ttft_avg': float,
            'itl_avg': float,
            'e2e_avg': float,
            'model_id': str,
            'gpu_id': str,
        }
        """
        df = self._dict_to_df(model_name, features_dict)
        X = self.preproc.transform(df)
        with torch.no_grad():
            Xv = torch.tensor(X, dtype=torch.float32).to(self.device)
            ttft_log, tpot_log = self.model(Xv)
            ttft = float(np.exp(ttft_log.cpu().numpy().squeeze()))
            tpot = float(np.exp(tpot_log.cpu().numpy().squeeze()))
        return ttft, tpot

    def _dict_to_df(self, model_name, feat):
        import pandas as pd
        d = {
            **feat,
            "model_gpu": f"{model_name}_{feat['gpu_id']}"
        }
        return pd.DataFrame([d])
