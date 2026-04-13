import torch
from pilot.model.u_net import UNet


def test_unet_output_shape():
    model = UNet(7, 1)
    x = torch.randn(2, 7, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64), f"Expected (2, 1, 64, 64), got {out.shape}"


def test_unet_no_nan():
    model = UNet(7, 1)
    x = torch.randn(1, 7, 64, 64)
    out = model(x)
    assert not torch.isnan(out).any(), "Model output contains NaN"


def test_unet_gradient_flow():
    model = UNet(7, 1)
    x = torch.randn(1, 7, 64, 64)
    out = model(x)
    loss = out.mean()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


if __name__ == "__main__":
    test_unet_output_shape()
    test_unet_no_nan()
    test_unet_gradient_flow()
    print("All tests passed.")
