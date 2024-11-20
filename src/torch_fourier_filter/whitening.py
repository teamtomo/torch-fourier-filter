"""Noise whitening functions."""

import einops
import torch
import torch.nn.functional as F

from torch_fourier_filter.dft_utils import (
    _1d_to_rotational_average_2d_dft,
    _1d_to_rotational_average_3d_dft,
    rotational_average_dft_2d,
    rotational_average_dft_3d,
)


def gaussian_smoothing(
    tensor: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0
) -> torch.Tensor:
    """
    Apply Gaussian smoothing to a 1D tensor.

    Parameters
    ----------
    tensor: torch.Tensor
        The input tensor to be smoothed.
    kernel_size: int
        The size of the Gaussian kernel.
    sigma: float
        The standard deviation of the Gaussian kernel.

    Returns
    -------
    torch.Tensor
        The smoothed tensor.
    """
    # Create a 1D Gaussian kernel
    x = torch.arange(
        -kernel_size // 2 + 1,
        kernel_size // 2 + 1,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # Debug: Print the kernel
    print("Gaussian kernel:", kernel)

    # Reshape kernel for convolution
    kernel = einops.rearrange(kernel, "k -> 1 1 k")

    # Apply the Gaussian kernel to the tensor
    tensor = einops.rearrange(tensor, "n -> 1 1 n")  # Add batch and channel dimensions
    smoothed_tensor = F.conv1d(tensor, kernel, padding=kernel_size // 2)

    # Debug: Print the smoothed tensor
    print("Smoothed tensor:", smoothed_tensor.view(-1))

    return smoothed_tensor.view(-1)


def whitening_filter(
    image_dft: tuple[int, int] | tuple[int, int, int],
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool = True,
    fftshift: bool = False,
    dimensions_output: int = 2,  # 1/2/3D filter
    smoothing: bool = False,
    power_spec: bool = True,
) -> torch.tensor:
    """
    Create a whitening filter from an image DFT.

    Parameters
    ----------
    image_dft: torch.Tensor
        The DFT of the image.
    image_shape: tuple[int, ...]
        Shape of the input image
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False)
    fftshift: bool
        Whether the input is fftshifted
    dimensions_output: int
        The number of dimensions of the output filter (1/2/3)
    smoothing: bool
        Whether to apply Gaussian smoothing to the filter
    power_spec: bool
        Whether to use the power spectrum instead or the amplitude spectrum

    Returns
    -------
    torch.Tensor
        Whitening filter
    """
    power_spectrum = torch.absolute(image_dft)
    if power_spec:
        power_spectrum = power_spectrum**2
    radial_average = None
    if len(image_shape) == 2:
        radial_average, _ = rotational_average_dft_2d(
            dft=power_spectrum,
            image_shape=image_shape,
            rfft=rfft,
            fftshifted=fftshift,
            return_2d_average=False,  # output 1D average
        )
    elif len(image_shape) == 3:
        radial_average, _ = rotational_average_dft_3d(
            dft=power_spectrum,
            image_shape=image_shape,
            rfft=rfft,
            fftshifted=fftshift,
            return_3d_average=False,  # output 1D average
        )

    # Take the reciprical of the square root of the radial average
    whiten_filter = 1 / (radial_average)
    if power_spec:
        whiten_filter = whiten_filter**0.5

    # Apply Gaussian smoothing
    if smoothing:
        whiten_filter = gaussian_smoothing(whiten_filter)

    # put back to 2 or 3D if necessary
    if dimensions_output == 2:
        if len(power_spectrum.shape) > len(image_shape):
            image_shape = (*power_spectrum.shape[:-2], *image_shape[-2:])
        whiten_filter = _1d_to_rotational_average_2d_dft(
            values=radial_average,
            image_shape=image_shape,
            rfft=rfft,
            fftshifted=fftshift,
        )
    elif dimensions_output == 3:
        if len(power_spectrum.shape) > len(image_shape):
            image_shape = (*power_spectrum.shape[:-3], *image_shape[-3:])
        whiten_filter = _1d_to_rotational_average_3d_dft(
            values=radial_average,
            image_shape=image_shape,
            rfft=rfft,
            fftshifted=fftshift,
        )

    # Normalize the filter
    whiten_filter /= whiten_filter.max()

    return whiten_filter
