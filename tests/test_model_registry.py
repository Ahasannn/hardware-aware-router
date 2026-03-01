"""Unit tests for hw_router.model_registry."""

import pytest

from hw_router.model_registry import get_model_id, get_model_hugging_face_name, get_all_models


class TestGetModelId:
    def test_known_models(self):
        assert get_model_id("qwen14b") == 0
        assert get_model_id("phi3-mini") == 1
        assert get_model_id("llama3-8b") == 2
        assert get_model_id("qwen3b") == 3
        assert get_model_id("mistral7b") == 4

    def test_path_resolution(self):
        assert get_model_id("/home/user/models/qwen14b") == 0
        assert get_model_id("/any/path/to/phi3-mini") == 1

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_model_id("unknown-model")


class TestGetModelHuggingFaceName:
    def test_known_models(self):
        assert get_model_hugging_face_name("qwen14b") == "Qwen2.5-14B-Instruct"
        assert get_model_hugging_face_name("llama3-8b") == "Llama-3.1-8B-Instruct"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_model_hugging_face_name("nonexistent")


class TestGetAllModels:
    def test_returns_all_five(self):
        models = get_all_models()
        assert len(models) == 5

    def test_tuple_format(self):
        models = get_all_models()
        for basename, model_id, hf_name in models:
            assert isinstance(basename, str)
            assert isinstance(model_id, int)
            assert isinstance(hf_name, str)

    def test_ids_are_sequential(self):
        models = get_all_models()
        ids = [mid for _, mid, _ in models]
        assert ids == [0, 1, 2, 3, 4]
