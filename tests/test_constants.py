"""Unit tests for hw_router.constants."""

from hw_router.constants import (
    HF_MODEL_NAMES,
    MODEL_PRICES,
    MODEL_QUALITY,
    DEFAULT_LAMBDA,
    LAT_P95_LOG,
    STATIC_COST_P95,
)


class TestModelPrices:
    def test_all_models_have_prices(self):
        for model in HF_MODEL_NAMES:
            assert model in MODEL_PRICES, f"Missing price for {model}"

    def test_prices_are_positive(self):
        for model, price in MODEL_PRICES.items():
            assert price > 0, f"Price for {model} should be positive"


class TestModelQuality:
    def test_all_models_have_quality(self):
        for model in HF_MODEL_NAMES:
            assert model in MODEL_QUALITY, f"Missing quality for {model}"

    def test_quality_in_range(self):
        for model, quality in MODEL_QUALITY.items():
            assert 0.0 <= quality <= 1.0, f"Quality for {model} should be in [0, 1]"


class TestConstants:
    def test_default_lambda_in_range(self):
        assert 0.0 <= DEFAULT_LAMBDA <= 1.0

    def test_normalization_constants_positive(self):
        assert LAT_P95_LOG > 0
        assert STATIC_COST_P95 > 0
