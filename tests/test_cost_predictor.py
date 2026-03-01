"""Unit tests for hw_router.cost_predictor (no checkpoints needed)."""

import pytest
import torch

from hw_router.cost_predictor import HardwareCostNet


class TestHardwareCostNet:
    def test_forward_shape(self):
        net = HardwareCostNet(input_dim=20)
        x = torch.randn(4, 20)
        ttft, tpot = net(x)
        assert ttft.shape == (4, 1)
        assert tpot.shape == (4, 1)

    def test_single_sample(self):
        net = HardwareCostNet(input_dim=10)
        x = torch.randn(1, 10)
        ttft, tpot = net(x)
        assert ttft.shape == (1, 1)
        assert tpot.shape == (1, 1)

    def test_output_is_finite(self):
        net = HardwareCostNet(input_dim=15)
        x = torch.randn(8, 15)
        ttft, tpot = net(x)
        assert torch.isfinite(ttft).all()
        assert torch.isfinite(tpot).all()

    def test_exp_output_is_positive(self):
        """Predictions are in log-space; exp should be positive."""
        net = HardwareCostNet(input_dim=10)
        x = torch.randn(4, 10)
        ttft, tpot = net(x)
        assert (torch.exp(ttft) > 0).all()
        assert (torch.exp(tpot) > 0).all()

    def test_different_input_dims(self):
        for dim in [5, 10, 20, 50]:
            net = HardwareCostNet(input_dim=dim)
            x = torch.randn(2, dim)
            ttft, tpot = net(x)
            assert ttft.shape == (2, 1)

    def test_gradients_flow(self):
        net = HardwareCostNet(input_dim=10)
        x = torch.randn(2, 10, requires_grad=True)
        ttft, tpot = net(x)
        loss = ttft.sum() + tpot.sum()
        loss.backward()
        assert x.grad is not None
