# model_maps.py

LOCAL_MODEL_TO_ID = {
    "/home/ah872032/models/qwen14b" : 0,
    "/home/ah872032/models/phi3-mini" : 1,
    "/home/ah872032/models/llama3-8b" : 2,
    "/home/ah872032/models/qwen3b" : 3,
    "/home/ah872032/models/mistral7b": 4,
}


LOCAL_MODEL_TO_HUGGINGFACE_NAME = {
    "/home/ah872032/models/qwen14b" : "Qwen2.5-14B-Instruct",
    "/home/ah872032/models/phi3-mini" : "Phi-3-mini-128k-instruct",
    "/home/ah872032/models/llama3-8b" : "Llama-3.1-8B-Instruct",
    "/home/ah872032/models/qwen3b" : "Qwen2.5-3B-Instruct",
    "/home/ah872032/models/mistral7b" : "Mistral-7B-Instruct-v0.3",
}


def get_model_id(name: str) -> int:
    if name in LOCAL_MODEL_TO_ID:
        return LOCAL_MODEL_TO_ID[name]
    raise KeyError(f"Unknown model name: {name}")

def get_model_hugging_face_name(name: str) -> str:
    if name in LOCAL_MODEL_TO_HUGGINGFACE_NAME:
        return LOCAL_MODEL_TO_HUGGINGFACE_NAME[name]
    raise KeyError(f"Unknown model name: {name}")
