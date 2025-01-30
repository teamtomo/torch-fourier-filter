"""Noise whitening functions."""

import einops
import torch
import torch.nn.functional as F
from torch_grid_utils.fftfreq_grid import fftfreq_grid


def expand_1d_to_ndim_spectrum(
    spectrum_1d: torch.Tensor,
    freq_grid: torch.Tensor,
    unique_freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Expand a 1D spectrum back to N-D based on frequency grid values.

    Parameters
    ----------
    spectrum_1d: torch.Tensor
        1D spectrum values, shape (num_freqs,) or (batch, num_freqs)
    freq_grid: torch.Tensor
        nD grid of frequency values to map on to, shape (*spatial_dims)
    unique_freqs: torch.Tensor
        Unique frequency values, shape (num_freqs,)

    Returns
    -------
    torch.Tensor
        N-D spectrum with same shape as freq_grid (or batch, *freq_grid.shape)
    """
    # Create an output tensor with the same batch dimensions and shape as freq_grid
    spectrum_ndim = torch.zeros(
        (*spectrum_1d.shape[:-1], *freq_grid.shape), device=spectrum_1d.device
    )

    # Iterate over unique frequencies and apply them to the corresponding positions
    for i, freq in enumerate(unique_freqs):
        mask = freq_grid == freq
        spectrum_ndim[..., mask] = spectrum_1d[..., i].unsqueeze(-1)

    return spectrum_ndim


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


def whitening_filter(
    image_dft: torch.Tensor,
    image_shape: tuple[int, int] | tuple[int, int, int],
    output_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool = True,
    fftshift: bool = False,
    dim: int | tuple[int, ...] = (-2, -1),  # output dimensions
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
        Shape of the output filter (in real space)
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False)
    fftshift: bool
        Whether the input is fftshifted
    dim: int | tuple[int, ...]
        Output dimension axes (excluding batch dimension if present)
    smoothing: bool
        Whether to apply Gaussian smoothing to the filter
    power_spec: bool
        Whether to use the power spectrum instead or the amplitude spectrum

    Returns
    -------
    torch.Tensor
        Whitening filter
    """
    ###Move batch dimension to front if not already there###

    # Convert single dim to tuple
    if isinstance(dim, int):
        dim = (dim,)
    # Convert negative dims to positive
    dim = tuple(d if d >= 0 else image_dft.ndim + d for d in dim)

    # Move all non-spatial (batch) dimensions to the front
    batch_dims = [i for i in range(image_dft.ndim) if i not in dim]
    if batch_dims != list(range(len(batch_dims))):
        image_dft = image_dft.permute(*batch_dims, *dim)

    power_spectrum = torch.abs(image_dft)

    if power_spec:
        power_spectrum = power_spectrum**2

    # output shape is for the image
    # Handle rfft case for output shape
    if rfft:
        last_dim = output_shape[-1] // 2 + 1
        output_shape_fft = (*output_shape[:-1], last_dim)
    else:
        output_shape_fft = output_shape

    # Add batch dimension if necessary

    # Interpolate or bin the power spectrum
    if (
        power_spectrum.shape[-(len(output_shape)) :]
        != output_shape_fft[-(len(output_shape)) :]
    ):
        power_spectrum = F.interpolate(
            power_spectrum.unsqueeze(1),  # add channel dim
            size=output_shape_fft[-(len(output_shape)) :],  # spatial dims only
            mode="bilinear" if len(output_shape) == 2 else "trilinear",
            align_corners=False,
        ).squeeze(1)  # remove channel dim

    # Get FFT freqs
    freq_grid = fftfreq_grid(
        image_shape=output_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=power_spectrum.device,
    )

    # Bin frequencies and average power spectrum values to 1D
    unique_freqs = torch.unique(freq_grid)
    binned_spectrum_1d = torch.zeros(
        (*power_spectrum.shape[: len(batch_dims)], len(unique_freqs)),
        device=power_spectrum.device,
    )
    # Precompute the dimensions to sum over
    sum_dims = tuple(range(-len(dim), 0))

    # Vectorized binning
    for i, freq in enumerate(unique_freqs):
        mask = freq_grid == freq
        masked_power = power_spectrum * mask
        binned_spectrum_1d[..., i] = masked_power.sum(dim=sum_dims) / mask.sum(
            dim=sum_dims
        )

    power_spectrum = expand_1d_to_ndim_spectrum(
        binned_spectrum_1d, freq_grid, unique_freqs
    )

    whitening_filter = 1 / power_spectrum

    if power_spec:
        whitening_filter = whitening_filter**0.5

    # Apply Gaussian smoothing
    if smoothing:
        whitening_filter = gaussian_smoothing(
            tensor=whitening_filter,
            dim=dim,
        )

    # Normalize the filter
    if len(whitening_filter.shape) > len(output_shape):  # then batched
        whitening_filter_max = whitening_filter.amax(dim=dim, keepdim=True)
        whitening_filter /= whitening_filter_max
    else:
        whitening_filter /= whitening_filter.max()

    return whitening_filter
