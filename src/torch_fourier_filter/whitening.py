"""Noise whitening functions."""

from typing import Optional

import einops
import torch
import torch.nn.functional as F
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from torch_fourier_filter.utils import bin_1dim_with_lerp, curve_1dim_to_ndim


def gaussian_smoothing(
    tensor: torch.Tensor,
    dim: int | tuple[int, ...] = -1,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Apply Gaussian smoothing over specified dimensions of a tensor.

    Parameters
    ----------
    tensor: torch.Tensor
        The input tensor to be smoothed.
    dim: int | tuple[int, ...]
        Dimensions over which to apply smoothing. Can be a single int or tuple of ints.
        Negative dimensions are indexed from the end.
    kernel_size: int
        The size of the Gaussian kernel.
    sigma: float
        The standard deviation of the Gaussian kernel.

    Returns
    -------
    torch.Tensor
        The smoothed tensor with same shape as input.
    """
    # Convert single dim to tuple
    if isinstance(dim, int):
        dim = (dim,)

    # Convert negative dims to positive
    dim = tuple(d if d >= 0 else tensor.dim() + d for d in dim)

    # Validate dimensions
    if not all(0 <= d < tensor.dim() for d in dim):
        raise ValueError(f"Invalid dimensions {dim} for tensor of rank {tensor.dim()}")
    if len(dim) > 2:
        raise ValueError("Gaussian smoothing only supports 1D or 2D operations")

    # Create coordinate grid for kernel
    x = torch.arange(
        -kernel_size // 2 + 1,
        kernel_size // 2 + 1,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    if len(dim) == 1:  # 1D smoothing
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()

        # Create conv kernel with singleton dimensions
        shape = [1] * (tensor.dim())
        shape[dim[0]] = kernel_size
        kernel = kernel.view(*shape)

        # Apply 1D convolution
        tensor = einops.rearrange(tensor, "... n -> ... 1 1 n")
        kernel = einops.rearrange(kernel, "... n -> ... 1 1 n")
        smoothed_tensor = F.conv1d(tensor, kernel, padding=kernel_size // 2)
        return einops.rearrange(smoothed_tensor, "... 1 1 n -> ... n")

    else:  # 2D smoothing
        x, y = torch.meshgrid(x, x, indexing="ij")
        kernel = torch.exp(-0.5 * ((x / sigma) ** 2 + (y / sigma) ** 2))
        kernel = kernel / kernel.sum()

        # Create conv kernel with singleton dimensions
        shape = [1] * (tensor.dim())
        shape[dim[0]] = kernel_size
        shape[dim[1]] = kernel_size
        kernel = kernel.view(*shape)
        kernel = einops.rearrange(kernel, "... h w -> ... 1 1 h w")
        # Apply 2D convolution
        tensor = einops.rearrange(tensor, "... h w -> ... 1 1 h w")
        smoothed_tensor = F.conv2d(
            tensor, kernel, padding=kernel_size // 2, stride=(1, 1)
        )
        return einops.rearrange(smoothed_tensor, "... 1 1 h w -> ... h w")


def calculate_num_freq_bins(image_shape: tuple[int, ...]) -> int:
    """Calculate the number of frequency bins for a power spectrum given image shape.

    Parameters
    ----------
    image_shape: tuple[int, ...]
        Shape of the image.

    Returns
    -------
    int
        Number of frequency bins.
    """
    largest_dim = max(image_shape)

    return int((largest_dim // 2 + 1) * len(image_shape) ** 0.5 + 1)


def power_spectral_density(
    image_dft: torch.Tensor,
    rfft: bool = True,
    fftshift: bool = False,
    dim: int | tuple[int, ...] = (-2, -1),
    num_freqs: Optional[int] = None,
    do_power_spectrum: Optional[bool] = True,
) -> torch.Tensor:
    """Calculates the power spectral density of an image.

    Note that the power spectrum can either be calculated over the intensities or
    amplitudes of the Fourier components; the parameter `do_power_spectrum` controls
    this.

    Parameters
    ----------
    image_dft: torch.Tensor
        The DFT of the images/volumes.
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False).
    fftshift: bool
        Whether the input is fftshifted.
    dim: int | tuple[int, ...], optional
        Which dimensions to calculate the power spectrum over. The default is (-2, -1)
        which corresponds to the spatial dimensions of an image.
    num_freqs: int, optional
        Number of frequency bins to calculate the power spectrum over. If None, the
        the number is inferred from the input shape automatically.
    do_power_spectrum: bool, optional
        Weather to use the power (intensity) spectrum or the amplitude spectrum. If
        True, the power spectrum is calculated, otherwise the amplitude spectrum is
        calculated. The default is True.

    Returns
    -------
    torch.Tensor
        The power spectral density of the input image. Includes batch dimensions, if
        present.

    Raises
    ------
    ValueError
        Currently *does not* support batching. Raises error if the input image is larger
        than the number of dimensions.
    """
    if isinstance(dim, int):
        dim = (dim,)
    if image_dft.ndim > len(dim):
        raise ValueError("Batched power spectral density not supported yet.")

    # convert dims to positive and tuple
    if isinstance(dim, int):
        dim = (dim,)
    dim = tuple(d if d >= 0 else image_dft.ndim + d for d in dim)

    # compute real-space shape from DFT shape and rfft argument
    real_space_shape = [s for i, s in enumerate(image_dft.shape) if i in dim]
    if rfft:
        last_dim = 2 * (real_space_shape[-1] - 1)
        real_space_shape[-1] = last_dim

    # Construct 1-dimensional grid of frequencies for binning
    if num_freqs is None:
        num_freqs = calculate_num_freq_bins(tuple(real_space_shape))
    binning_freqs = torch.linspace(
        start=0.0,
        end=(len(dim) ** 0.5) / 2,
        steps=num_freqs,
        device=image_dft.device,
    )

    power_spectrum = torch.abs(image_dft)

    if do_power_spectrum:
        power_spectrum = power_spectrum**2

    # Construct FFT frequency grid
    freq_grid = fftfreq_grid(
        image_shape=real_space_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=image_dft.device,
    )

    # Bin the power spectrum into 1-dimension
    power_spectrum_1d = bin_1dim_with_lerp(
        x=freq_grid,
        y=power_spectrum,
        xp=binning_freqs,
        normalize_by_count=True,
    )

    if do_power_spectrum:
        power_spectrum_1d = torch.sqrt(power_spectrum_1d)

    return power_spectrum_1d


def whitening_filter(
    image_dft: torch.Tensor,
    rfft: bool = True,
    fftshift: bool = False,
    dim: int | tuple[int, ...] = (-2, -1),
    num_freqs: Optional[int] = None,
    do_power_spectrum: bool = True,
    output_shape: Optional[tuple[int, ...]] = None,
    output_rfft: Optional[bool] = None,
    output_fftshift: Optional[bool] = None,
) -> torch.Tensor:
    """Create a whitening filter the discrete Fourier transform of an input.

    Parameters
    ----------
    image_dft: torch.Tensor
        The DFT of the images/volumes.
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False).
    fftshift: bool
        Whether the input is fftshifted.
    dim: int | tuple[int, ...], optional
        Which dimensions to calculate the power spectrum over. The default is (-2, -1)
        which corresponds to the spatial dimensions of an image.
    num_freqs: int, optional
        Number of frequency bins to calculate the power spectrum over. If None, the
        the number is inferred from the input shape automatically.
    do_power_spectrum: bool, optional
        Weather to use the power (intensity) spectrum or the amplitude spectrum. If
        True, the power spectrum is calculated, otherwise the amplitude spectrum is
        calculated. The default is True.
    output_shape: tuple[int, ...], optional
        The shape of the output filter. If None, then the shape is the same as
        `image_dft`. The default is None. NOTE: This shape is in Fourier space.
    output_rfft: bool, optional
        Whether to return the filter in rfft form. If None, then the same as `rfft`.
        The default is None.
    output_fftshift: bool, optional
        Whether to return the filter fftshifted. If None, then the same as `fftshift`.
        The default is None.

    Returns
    -------
    torch.Tensor
        The whitening filter in Fourier space.
    """
    # convert dims to positive and tuple
    if isinstance(dim, int):
        dim = (dim,)
    dim = tuple(d if d >= 0 else image_dft.ndim + d for d in dim)

    power_spec_1d = power_spectral_density(
        image_dft=image_dft,
        rfft=rfft,
        fftshift=fftshift,
        dim=dim,
        num_freqs=num_freqs,
        do_power_spectrum=do_power_spectrum,
    )

    # Create the filter
    whitening_filter_1d = torch.where(
        power_spec_1d != 0, 1 / power_spec_1d, torch.zeros_like(power_spec_1d)
    )
    whitening_filter_1d /= whitening_filter_1d.max()

    # Construct FFT frequency grid
    if output_shape is None:
        output_shape = image_dft.shape

    if output_rfft is None:
        output_rfft = rfft

    if output_fftshift is None:
        output_fftshift = fftshift

    real_space_shape = [s for i, s in enumerate(output_shape) if i in dim]
    if output_rfft:
        last_dim = 2 * (real_space_shape[-1] - 1)
        real_space_shape[-1] = last_dim

    freq_grid = fftfreq_grid(
        image_shape=tuple(real_space_shape),
        rfft=output_rfft,
        fftshift=output_fftshift,
        norm=True,
        device=image_dft.device,
    )

    # Interpolate the filter to the frequency grid
    last_freq = (len(dim) ** 0.5) / 2
    whitening_filter_ndim = curve_1dim_to_ndim(
        frequency_1d=torch.linspace(0, last_freq, steps=len(whitening_filter_1d)),
        values_1d=whitening_filter_1d,
        frequency_grid=freq_grid,
        fill_lower=0.0,  # Fill oob areas with zero (sets frequencies to zero)
        fill_upper=0.0,
    )

    return whitening_filter_ndim
