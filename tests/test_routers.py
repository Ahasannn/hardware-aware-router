"""Unit tests for hw_router.routers (no model loading needed)."""

import pytest

from hw_router.routers import BaseRouter, BaselineRouter, RandomRouter, RoundRobinRouter


class TestBaselineRouter:
    def test_returns_tuple(self):
        router = BaselineRouter()
        result = router.compute("Qwen2.5-14B-Instruct", "test prompt")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_quality_in_range(self):
        router = BaselineRouter()
        quality, cost = router.compute("Qwen2.5-14B-Instruct", "test prompt")
        assert 0.0 <= quality <= 1.0

    def test_cost_is_zero(self):
        router = BaselineRouter()
        _, cost = router.compute("Qwen2.5-14B-Instruct", "test prompt")
        assert cost == 0.0

    def test_unknown_model_returns_default(self):
        router = BaselineRouter()
        quality, _ = router.compute("unknown-model", "test")
        assert quality == 1.0  # default fallback


class TestRandomRouter:
    def test_returns_tuple(self):
        router = RandomRouter()
        result = router.compute("any-model", "any prompt")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_values_in_range(self):
        router = RandomRouter()
        for _ in range(50):
            quality, cost = router.compute("model", "prompt")
            assert 0.0 <= quality <= 1.0
            assert 0.0 <= cost <= 1.0

    def test_values_vary(self):
        router = RandomRouter()
        qualities = [router.compute("m", "p")[0] for _ in range(20)]
        assert len(set(qualities)) > 1  # not all the same


class TestRoundRobinRouter:
    def test_returns_tuple(self):
        router = RoundRobinRouter()
        result = router.compute("Qwen2.5-14B-Instruct", "test")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cost_is_zero(self):
        router = RoundRobinRouter()
        _, cost = router.compute("Qwen2.5-14B-Instruct", "test")
        assert cost == 0.0


class TestBaseRouterInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseRouter()

    def test_subclass_must_implement_compute(self):
        class IncompleteRouter(BaseRouter):
            pass

        with pytest.raises(TypeError):
            IncompleteRouter()

    def test_custom_subclass_works(self):
        class MyRouter(BaseRouter):
            def compute(self, model_name, prompt):
                return 0.5, 0.1

        router = MyRouter()
        q, c = router.compute("model", "prompt")
        assert q == 0.5
        assert c == 0.1
