"""Noise whitening functions."""

import einops
import torch
import torch.nn.functional as F


def gaussian_smoothing(
    tensor: torch.Tensor,
    spatial_dims: int = 1,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Apply Gaussian smoothing to a 1D or 2D tensor or batch.

    Parameters
    ----------
    tensor: torch.Tensor
        The input tensor to be smoothed. Can be:
        - 1D: (N,)
        - 2D: (H, W)
        - Batched 1D: (B, N)
        - Batched 2D: (B, H, W)
    kernel_size: int
        The size of the Gaussian kernel.
    sigma: float
        The standard deviation of the Gaussian kernel.
    spatial_dims: int
        Number of spatial dimensions (1 or 2)

    Returns
    -------
    torch.Tensor
        The smoothed tensor with same shape as input.
    """
    assert spatial_dims in [1, 2], "spatial_dims must be 1 or 2"
    dim_input_tensor = tensor.dim()
    is_batched = dim_input_tensor > spatial_dims

    # Create coordinate grid for kernel
    x = torch.arange(
        -kernel_size // 2 + 1,
        kernel_size // 2 + 1,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    if spatial_dims == 1:
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = einops.rearrange(kernel, "k -> 1 1 k")

        # Reshape input
        if is_batched:
            tensor = einops.rearrange(tensor, "b n -> b 1 n")
        else:
            tensor = einops.rearrange(tensor, "n -> 1 1 n")

        # Apply smoothing
        smoothed = F.conv1d(tensor, kernel, padding=kernel_size // 2)

        # Reshape output
        if is_batched:
            return einops.rearrange(smoothed, "b 1 n -> b n")
        else:
            return einops.rearrange(smoothed, "1 1 n -> n")

    else:  # 2D case
        x, y = torch.meshgrid(x, x, indexing="ij")
        kernel = torch.exp(-0.5 * ((x / sigma) ** 2 + (y / sigma) ** 2))
        kernel = kernel / kernel.sum()
        kernel = einops.rearrange(kernel, "h w -> 1 1 h w")

        # Reshape input
        if is_batched:
            tensor = einops.rearrange(tensor, "b h w -> b 1 h w")
        else:
            tensor = einops.rearrange(tensor, "h w -> 1 1 h w")

        # Apply smoothing
        smoothed = F.conv2d(tensor, kernel, padding=kernel_size // 2)

        # Reshape output
        if is_batched:
            return einops.rearrange(smoothed, "b 1 h w -> b h w")
        else:
            return einops.rearrange(smoothed, "1 1 h w -> h w")


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
        Shape of the output filter (in real space)
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

    # output shape is for the image
    # Handle rfft case for output shape
    if rfft:
        last_dim = output_shape[-1] // 2 + 1
        output_shape_fft = (*output_shape[:-1], last_dim)
    else:
        output_shape_fft = output_shape

    # Add batch dimension if necessary
    if len(power_spectrum.shape) > len(output_shape):  # batched case
        batch_size = power_spectrum.shape[0]
        resized_shape = (batch_size, *output_shape_fft)
    else:
        resized_shape = output_shape_fft

    # Interpolate or bin the power spectrum
    if (
        power_spectrum.shape[-(len(output_shape)) :]
        != resized_shape[-(len(output_shape)) :]
    ):
        power_spectrum = F.interpolate(
            power_spectrum.unsqueeze(1),  # add channel dim
            size=resized_shape[-(len(output_shape)) :],  # spatial dims only
            mode="bilinear" if len(output_shape) == 2 else "trilinear",
            align_corners=False,
        ).squeeze(1)  # remove channel dim

    whitening_filter = 1 / power_spectrum

    if power_spec:
        whitening_filter = whitening_filter**0.5

    # Apply Gaussian smoothing
    if smoothing:
        whitening_filter = gaussian_smoothing(
            tensor=whitening_filter,
            spatial_dims=dimensions_output,
        )

    # Normalize the filter
    if len(whitening_filter.shape) > dimensions_output:  # then batched
        whitening_filter_max = einops.reduce(whitening_filter, "b ... -> b", "max")
        whitening_filter_max = einops.rearrange(whitening_filter_max, "b -> b 1 1")
        whitening_filter /= whitening_filter_max
    else:
        whitening_filter /= whitening_filter.max()

    return whitening_filter
