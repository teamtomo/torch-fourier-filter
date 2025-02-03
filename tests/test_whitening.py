import torch

from torch_fourier_filter.whitening import gaussian_smoothing, whitening_filter


def test_whitening_filter():
    # Test case 1: Basic functionality with 2D input
    image_shape = (8, 8)
    image = torch.rand(image_shape, dtype=torch.float32)
    image_dft = torch.fft.fft2(image)

    result = whitening_filter(
        image_dft,
        image_shape,
        image_shape,
        rfft=False,
        fftshift=False,
        dimensions_output=2,
        smoothing=False,
    )
    assert torch.isfinite(
        result
    ).all(), "Test case 1 failed: Non-finite values in result"
    assert result.max() == 1.0, "Test case 1 failed: Result not normalized"

    # Test case 2: Basic functionality with 3D input
    image_shape = (16, 16, 16)
    image = torch.rand(image_shape, dtype=torch.float32)
    image_dft = torch.fft.fftn(image, dim=(-3, -2, -1))

    result = whitening_filter(
        image_dft,
        image_shape,
        image_shape,
        rfft=False,
        fftshift=False,
        dimensions_output=3,
        smoothing=False,
    )
    assert result.shape == image_shape, "Test case 2 failed: Shape mismatch"
    assert torch.isfinite(
        result
    ).all(), "Test case 2 failed: Non-finite values in result"
    assert result.max() == 1.0, "Test case 2 failed: Result not normalized"

    # Test case 3: With smoothing
    image_shape = (32, 32)
    image = torch.rand(image_shape, dtype=torch.float32)
    image_dft = torch.fft.fft2(image)

    result = whitening_filter(
        image_dft,
        image_shape,
        image_shape,
        rfft=False,
        fftshift=False,
        dimensions_output=2,
        smoothing=True,
    )
    assert result.shape == image_shape, "Test case 3 failed: Shape mismatch"
    assert torch.isfinite(
        result
    ).all(), "Test case 3 failed: Non-finite values in result"
    assert result.max() == 1.0, "Test case 3 failed: Result not normalized"

    # Test case 4: Basic functionality with batched 2D input
    image_shape = (100, 16, 16)
    image = torch.rand(image_shape, dtype=torch.float32)
    image_dft = torch.fft.fftn(image, dim=(-2, -1))

    result = whitening_filter(
        image_dft,
        image_shape=image_shape[-2:],
        output_shape=image_shape[-2:],
        rfft=False,
        fftshift=False,
        dimensions_output=2,
        smoothing=False,
    )
    assert result.shape == image_shape, "Test case 4 failed: Shape mismatch"
    assert torch.isfinite(
        result
    ).all(), "Test case 4 failed: Non-finite values in result"
    assert result.max() == 1.0, "Test case 4 failed: Result not normalized"

    # Test case 5: Different input/output size
    image_shape = (100, 16, 16)
    output_shape = (8, 8)
    image = torch.rand(image_shape, dtype=torch.float32)
    image_dft = torch.fft.fftn(image, dim=(-2, -1))

    result = whitening_filter(
        image_dft,
        image_shape=image_shape[-2:],
        output_shape=output_shape,
        rfft=False,
        fftshift=False,
        dimensions_output=2,
        smoothing=False,
    )
    assert result.shape[-2:] == output_shape, "Test case 5 failed: Shape mismatch"
    assert torch.isfinite(
        result
    ).all(), "Test case 5 failed: Non-finite values in result"
    assert result.max() == 1.0, "Test case 5 failed: Result not normalized"


def test_gaussian_smoothing():
    # Create a test tensor
    tensor = torch.zeros(100)
    tensor[45:55] = 1.0  # Create a spike in the middle

    # Apply Gaussian smoothing
    smoothed_tensor = gaussian_smoothing(tensor, kernel_size=5, sigma=1.0)

    # Check that the smoothed tensor is different from the original around the spike
    assert not torch.equal(
        tensor[44:56], smoothed_tensor[44:56]
    ), "Smoothing not applied "
