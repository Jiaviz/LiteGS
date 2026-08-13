import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA renderer test")


def test_depth_to_normal_is_finite_and_differentiable():
    from litegs.render.surfel import _depth_to_normal

    depth = torch.ones((1, 1, 32, 48), device="cuda", requires_grad=True)
    projection = torch.eye(4, device="cuda")
    normal = _depth_to_normal(depth, projection)
    assert normal.shape == (3, 32, 48)
    assert torch.isfinite(normal).all()
    normal.square().mean().backward()
    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()


def test_official_cuda_surfel_extension_is_available():
    from diff_surfel_rasterization import GaussianRasterizer

    assert GaussianRasterizer is not None
