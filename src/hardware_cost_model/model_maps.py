# model_maps.py

# ----------------------------
# 1. LOCAL PATH → ID
# ----------------------------
LOCAL_MODEL_TO_ID = {
    "/home/ah872032/models/qwen14b": 0,
    "/home/ah872032/models/phi3-mini": 1,
    "/home/ah872032/models/llama3-8b": 2,
    "/home/ah872032/models/qwen3b": 3,
    "/home/ah872032/models/mistral7b": 4,
}

# ----------------------------
# 2. HF NAME → ID
# ----------------------------
HF_MODEL_TO_ID = {
    "Qwen/Qwen3-14B": 0,
    "microsoft/Phi-3-mini-128k-instruct": 1,
    "meta-llama/Llama-3.1-8B-Instruct": 2,
    "Qwen/Qwen2.5-3B-Instruct": 3,
    "mistralai/Mistral-7B-Instruct-v0.3": 4,
}

# ----------------------------
# 3. ID → LOCAL PATH (reverse)
# ----------------------------
ID_TO_LOCAL_MODEL = {
    model_id: local_path
    for local_path, model_id in LOCAL_MODEL_TO_ID.items()
}

# ----------------------------
# 4. ID → HF NAME (reverse)
# ----------------------------
ID_TO_HF_MODEL = {
    model_id: hf_name
    for hf_name, model_id in HF_MODEL_TO_ID.items()
}

# ----------------------------
# 5. Unified helper
# ----------------------------
def get_model_id(name: str) -> int:
    """Try both local and HF mappings."""
    if name in LOCAL_MODEL_TO_ID:
        return LOCAL_MODEL_TO_ID[name]
    if name in HF_MODEL_TO_ID:
        return HF_MODEL_TO_ID[name]
    raise KeyError(f"Unknown model name: {name}")
