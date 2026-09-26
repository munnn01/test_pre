"""Contract tests for a single shared V8 output and valid gradients."""
from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from src.models.v8_restorer import V8Restorer
from ops.v8_train import pixel_loss


def test_initial_model_is_identity_at_128() -> None:
    x = torch.rand(1, 3, 4, 128, 128)
    net = V8Restorer(channels=8)
    assert torch.equal(net(x, 35, 128), x)


def test_initial_model_upsamples_native_96_without_changing_time_order() -> None:
    x = torch.rand(1, 3, 4, 96, 96)
    net = V8Restorer(channels=8)
    y = net(x, 40, 96)
    expected = F.interpolate(x.permute(0, 2, 1, 3, 4).reshape(4, 3, 96, 96),
                             size=(128, 128), mode="bilinear", align_corners=False)
    expected = expected.reshape(1, 4, 3, 128, 128).permute(0, 2, 1, 3, 4)
    assert torch.equal(y, expected)


def test_loss_reaches_decoder_parameters() -> None:
    net = V8Restorer(channels=8)
    x = torch.rand(1, 3, 4, 128, 128)
    target = (x + .02).clamp(0, 1)
    loss = pixel_loss(net(x, 35, 128), target)
    loss.backward()
    assert net.head.weight.grad is not None
    assert bool(torch.isfinite(net.head.weight.grad).all())
    assert float(net.head.weight.grad.abs().sum()) > 0


def test_invalid_codec_condition_rejected() -> None:
    net = V8Restorer(channels=8)
    with pytest.raises(ValueError):
        net(torch.zeros(1, 3, 4, 128, 128), 60, 128)
