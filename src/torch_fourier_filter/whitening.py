"""Noise whitening functions."""

import einops
import torch
import torch.nn.functional as F

from torch_fourier_filter.dft_utils import (
    _1d_to_rotational_average_2d_dft,
    _1d_to_rotational_average_3d_dft,
    bin_or_interpolate_to_output_size,
    rotational_average_dft_2d,
    rotational_average_dft_3d,
)


def gaussian_smoothing(
    tensor: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0
) -> torch.Tensor:
    """
    Apply Gaussian smoothing to a 1D tensor or batch.

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

    # Reshape kernel for convolution
    kernel = einops.rearrange(kernel, "k -> 1 1 k")

    # Add batch and channel dimensions if necessary
    dim_input_tensor = tensor.dim()
    if dim_input_tensor == 1:
        tensor = einops.rearrange(tensor, "n -> 1 1 n")
    else:
        tensor = einops.rearrange(tensor, "b n -> b 1 n")

    # Apply the Gaussian kernel to the tensor
    smoothed_tensor = F.conv1d(tensor, kernel, padding=kernel_size // 2)

    if dim_input_tensor == 2:
        return einops.rearrange(smoothed_tensor, "b 1 n -> b n")
    else:
        return einops.rearrange(smoothed_tensor, "1 1 n -> n")


def whitening_filter(
    image_dft: torch.Tensor,
    image_shape: tuple[int, int] | tuple[int, int, int],
    output_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool = True,
    fftshift: bool = False,
    dimensions_output: int = 2,  # 1/2/3D filter
    smoothing: bool = False,
    power_spec: bool = True,
) -> torch.Tensor:
    """
    Create a whitening filter from an image DFT.

    Parameters
    ----------
    image_dft: torch.Tensor
        The DFT of the images/volumes
    image_shape: tuple[int, ...]
        Shape of the input image
    output_shape: tuple[int, ...]
        Shape of the output filter
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
    power_spectrum = torch.abs(image_dft)
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

    # bin or interpolate to output size
    whiten_filter = bin_or_interpolate_to_output_size(whiten_filter, output_shape)

    # put back to 2 or 3D if necessary
    if dimensions_output == 2:
        if len(power_spectrum.shape) > len(output_shape):
            output_shape = (*power_spectrum.shape[:-2], *output_shape[-2:])
        whiten_filter = _1d_to_rotational_average_2d_dft(
            values=radial_average,
            image_shape=output_shape,
            rfft=rfft,
            fftshifted=fftshift,
        )
    elif dimensions_output == 3:
        if len(power_spectrum.shape) > len(output_shape):
            output_shape = (*power_spectrum.shape[:-3], *output_shape[-3:])
        whiten_filter = _1d_to_rotational_average_3d_dft(
            values=radial_average,
            image_shape=output_shape,
            rfft=rfft,
            fftshifted=fftshift,
        )

    # Normalize the filter
    if len(whiten_filter.shape) > dimensions_output:  # then batched
        whiten_max = einops.reduce(whiten_filter, "b ... -> b", "max")
        whiten_max = einops.rearrange(whiten_max, "b -> b 1 1")
        whiten_filter /= whiten_max
    else:
        whiten_filter /= whiten_filter.max()

    return whiten_filter
