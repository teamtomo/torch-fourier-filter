"""Noise whitening functions."""

import einops
import torch
import torch.nn.functional as F

from torch_fourier_filter.dft_utils import (
    _1d_to_rotational_average_nd_dft,
    rotational_average_dft,
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

    # # Debug: Print the kernel
    # print("Gaussian kernel:", kernel)

    # Reshape kernel for convolution
    kernel = einops.rearrange(kernel, "k -> 1 1 k")

    # Apply the Gaussian kernel to the tensor
    tensor = einops.rearrange(tensor, "n -> 1 1 n")  # Add batch and channel dimensions
    smoothed_tensor = F.conv1d(tensor, kernel, padding=kernel_size // 2)

    # # Debug: Print the smoothed tensor
    # print("Smoothed tensor:", smoothed_tensor.view(-1))

    return smoothed_tensor.view(-1)


def whitening_filter_from_1d_power_spectrum(
    power_spectrum: torch.Tensor,
    filter_shape: tuple[int, ...],
    power_spec: bool = True,
    smoothing: bool = False,
    power_spectrum_pixel_size: float = 1.0,
    filter_pixel_size: float = 1.0,
    rfft: bool = False,
    fftshifted: bool = False,
) -> torch.Tensor:
    """Calculate a whitening filter for a known 1D power spectrum.

    NOTE: While the units for both the power spectrum and filter pixel sizes are
    defined in Angstroms, only their relative ratio is important for
    determining the 1D to 2D/3D mapping.

    NOTE: Different pixel sizes are not supported yet.

    Parameters
    ----------
    power_spectrum : torch.Tensor
        A 1D power spectrum.
    filter_shape : tuple[int, ...]
        The desired output shape of the filter in pixels. Either 2D or 3D.
    power_spec : bool
        Whether the input power spectrum is a power spectrum or amplitude spectrum.
    smoothing : bool
        Whether to apply Gaussian smoothing to the filter.
    power_spectrum_pixel_size : float
        The pixel size of the power spectrum in Angstroms.
    filter_pixel_size : float
        The pixel size of the filter in Angstroms.
    rfft : bool
        Whether the input is from an rfft (True) or full fft (False).
    fftshifted : bool
        Whether the input is fftshifted.

    Returns
    -------
    torch.Tensor
        The calculated whitening filter.
    """
    if power_spectrum_pixel_size != filter_pixel_size:
        raise NotImplementedError("Currently support only one pixel size.")

    # Take the reciprical of the square root of the radial average
    whiten_filter = 1 / (power_spectrum)
    if power_spec:
        whiten_filter = whiten_filter**0.5

    if smoothing:
        whiten_filter = gaussian_smoothing(whiten_filter)

    whiten_filter = _1d_to_rotational_average_nd_dft(
        values=whiten_filter,
        wanted_shape=filter_shape,
        rfft=rfft,
        fftshifted=fftshifted,
    )

    # Normalize the filter
    whiten_filter /= whiten_filter.max()

    return whiten_filter


def whitening_filter(
    image_dft: torch.Tensor,
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
    # Check the input parameters
    assert image_dft.ndim == 2 or image_dft.ndim == 3, "Input must be 2D or 3D"
    assert (
        len(image_shape) == 2 or len(image_shape) == 3
    ), "Image shape must be 2D or 3D"

    # Calculate power spectrum based on intensity or amplitude.
    power_spectrum = torch.absolute(image_dft)
    if power_spec:
        power_spectrum = power_spectrum**2

    rotational_average, frequency_bins = rotational_average_dft(
        dft=power_spectrum,
        image_shape=image_shape,
        rfft=rfft,
        fftshifted=fftshift,
        return_same_shape=False,  # output 1D average
    )

    whitening_filter = whitening_filter_from_1d_power_spectrum(
        power_spectrum=rotational_average,
        filter_shape=image_shape,
        power_spec=power_spec,
        smoothing=smoothing,
        power_spectrum_pixel_size=1.0,
        filter_pixel_size=1.0,
        rfft=rfft,
        fftshifted=fftshift,
    )

    return whitening_filter
