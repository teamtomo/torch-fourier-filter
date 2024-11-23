"""DFT utilities but currently only contains rotational averaging functions."""

from collections.abc import Sequence

import einops
import torch
from torch_grid_utils.coordinate_grid import coordinate_grid
from torch_grid_utils.fftfreq_grid import fftfreq_grid


def rotational_average_dft(
    dft: torch.Tensor,
    image_shape: tuple[int, ...],
    rfft: bool = False,
    fftshifted: bool = False,
    return_same_shape: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the rotational average of a 2D or 3D DFT.

    The returned frequency bins associated with the rotational average are in
    units of inverse pixels.

    Parameters
    ----------
    dft : torch.Tensor
        Complex tensor containing 2D or 3D Fourier transform(s). Can be batched
        with shape (batch, h, w) or unbatched (h, w) for 2D or (batch, d, h, w)
        or unbatched (d, h, w) for 3D.
    image_shape : tuple[int, int] | tuple[int, int, int]
        Shape of the input image / volume. Batching is supported.
    rfft : bool
        Whether the input is from an rfft (True) or full fft (False).
    fftshifted : bool
        Whether the input is fftshifted.
    return_same_shape : bool
        If true, the returned rotational average and frequency bins will have
        have the same shape as the input. Otherwise, the returned rotational
        average will be 1D.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Rotational average of the input DFT and the associated frequency bins.
    """
    _is_2d = len(image_shape) == 2
    _is_3d = len(image_shape) == 3
    _is_batched = len(dft.shape) > len(image_shape)  # TODO: check only +1 dim

    if _is_2d:
        shape = image_shape[-2:]
    elif _is_3d:
        shape = image_shape[-3:]
    else:
        raise ValueError("Only 2D and 3D images are supported.")

    # Calculate the number of frequency bins to include in the average
    n_bins = min((d // 2) + 1 for d in shape)

    # split data into frequency bins
    frequency_bins = _frequency_bin_centers(n_bins, device=dft.device)
    shell_data = _split_into_frequency_bins(
        dft=dft,
        n_bins=n_bins,
        image_shape=shape,
        rfft=rfft,
        fftshifted=fftshifted,
    )

    # calculate mean over each shell
    mean_per_shell = [
        einops.reduce(shell, "... shell -> ...", reduction="mean")
        for shell in shell_data
    ]
    rotational_average = einops.rearrange(mean_per_shell, "shells ... -> ... shells")

    # Broadcast the rotational average to the same shape as the input
    if return_same_shape:
        if _is_batched:
            image_shape = (*dft.shape[: -len(image_shape)], *image_shape)
        rotational_average = _1d_to_rotational_average_nd_dft(
            values=rotational_average,
            wanted_shape=image_shape,
            rfft=rfft,
            fftshifted=fftshifted,
            is_batched=_is_batched,
        )
        frequency_bins = _1d_to_rotational_average_nd_dft(
            values=frequency_bins,
            wanted_shape=image_shape,
            rfft=rfft,
            fftshifted=fftshifted,
            is_batched=_is_batched,
        )

    return rotational_average, frequency_bins


def _split_into_frequency_bins(
    dft: torch.Tensor,
    n_bins: int,
    image_shape: tuple[int, ...],
    rfft: bool = False,
    fftshifted: bool = False,
) -> list[torch.Tensor]:
    """Splits a DFT into frequency bins.

    Parameters
    ----------
    dft : torch.Tensor
        Complex tensor containing a 2D or 3D Fourier transform.
    n_bins : int
        Number of frequency bins to split the DFT into.
    image_shape : tuple[int, ...]
        Shape of the input image / volume without a batch dimension (if any).
    rfft : bool
        Whether the input is from an rfft (True) or full fft (False).
    fftshifted : bool
        Whether the input is fftshifted.

    Returns
    -------
    list[torch.Tensor]
        List of tensors containing the DFT values in each frequency bin.
    """
    _is_batched = len(dft.shape) > len(image_shape)  # TODO: check only +1 dims

    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshifted,
        norm=True,
        device=dft.device,
    )
    frequency_grid = frequency_grid.flatten()
    shell_borders = _frequency_bin_split_values(n_bins)
    shell_indices = _find_shell_indices_1d(frequency_grid, split_values=shell_borders)

    # Flatten the DFT (except along the first dimension if batched)
    # start_dim = 1 if _is_batched else 0
    # dft = dft.flatten(start_dim=start_dim)

    if dft.ndim == 2:
        dft = einops.rearrange(dft, "... h w -> ... (h w)")
    elif dft.ndim == 3:
        dft = einops.rearrange(dft, "... d h w -> ... (d h w)")

    shells = [dft[..., shell_idx] for shell_idx in shell_indices]

    return shells[:-1]


def _1d_to_rotational_average_nd_dft(
    values: torch.Tensor,
    wanted_shape: tuple[int, ...],
    rfft: bool = False,
    fftshifted: bool = False,
    is_batched: bool = False,
) -> torch.Tensor:
    """Map a set of 1-dimensional values onto an n-dimensional grid.

    Function assumes that the Fourier grid dimensions of the 1D values and
    the n-dimensional grid are the same (e.g. same sampling rate). Non-equal
    Fourier representations may be added in the future. The function also
    supports batched values of shape (batch, ...) if is_batched is True.

    Parameters
    ----------
    values : torch.Tensor
        1D tensor of values to map onto the grid
    wanted_shape : tuple[int, ...]
        Shape of the grid to map the values onto
    rfft : bool
        Weather the grid should be an rfft grid
    fftshifted : bool
        Weather the grid should be fftshifted
    is_batched : bool
        True if the first dimension represents a batched dimension

    Returns
    -------
    torch.Tensor
        The values mapped onto the grid
    """
    shape = wanted_shape[1:] if is_batched else wanted_shape
    shape = rfft_shape(shape) if rfft else shape

    _is_2d = len(shape) == 2
    _is_3d = len(shape) == 3

    # Construct the grid
    average_nd = torch.zeros(size=shape, dtype=values.dtype, device=values.device)

    # Construct the grid of frequencies
    grid = fftfreq_grid(
        image_shape=shape,
        rfft=rfft,
        fftshift=fftshifted,
        norm=True,
        device=values.device,
    )
    split_values = _frequency_bin_split_values(
        n=values.shape[-1],
        device=values.device,  # Why values.shape[-1]?
    )

    # Find the indices of the grid values in each shell
    shell_idx = _find_shell_indices(values=grid, split_values=split_values)
    shell_idx = shell_idx[:-1]

    # Iterate over each shell and insert the values
    for idx, shell in enumerate(shell_idx):
        idx_hwd = einops.rearrange(shell, "b idx -> idx b")
        average_nd[..., *(idx_hwd)] = values[..., [idx]]

    # fill outside the nyquist circle with the value from the nyquist bin
    average_nd[..., grid > 0.5] = values[..., [-1]]

    return average_nd


def _find_shell_indices(
    values: torch.Tensor, split_values: torch.Tensor
) -> list[torch.Tensor]:
    """
    Find n-dim indices which index to give values either side of split values.

    TODO: Describe further what this function returns (e.g. elements in each
    list are indices of values in each shell).
    """
    idx_nd = coordinate_grid(values.shape).long()
    idx_nd = idx_nd.flatten(end_dim=-2)  # Flatten except for last dim
    values = values.flatten()  # Flatten

    sorted_vals, sort_idx = torch.sort(values, descending=False)
    split_idx = torch.searchsorted(sorted_vals, split_values)

    # tensor_split requires the split_idx to live on cpu (why?)
    return list(torch.tensor_split(idx_nd[sort_idx], split_idx.to("cpu")))


def _find_shell_indices_1d(
    values: torch.Tensor, split_values: torch.Tensor
) -> list[torch.Tensor]:
    """
    Find indices which index to give values either side of split points.

    Parameters
    ----------
    values : torch.Tensor
        The values to split
    split_values : torch.Tensor
        The values to split at

    Returns
    -------
    list[torch.Tensor]
        List of tensors containing the indices of values in each shell
    """
    sorted_vals, sort_idx = torch.sort(values, descending=False)
    split_idx = torch.searchsorted(sorted_vals, split_values)
    # tensor_split requires the split_idx to live on cpu
    return list(torch.tensor_split(sort_idx, split_idx.to("cpu")))


def _frequency_bin_centers(n: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Centers of DFT sample frequency bins in units of inverse pixels.

    Parameters
    ----------
    n : int
        Number of frequency bins
    device : torch.device
        Device to place the tensor on

    Returns
    -------
    torch.Tensor
        Centers of the frequency bins
    """
    return torch.linspace(0, 0.5, steps=n, device=device)


def _frequency_bin_split_values(
    n: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Values at the borders of DFT sample frequency bins.

    Parameters
    ----------
    n : int
        Number of frequency bins
    device : torch.device
        Device to place the tensor on

    Returns
    -------
    torch.Tensor
        Borders of the frequency bins
    """
    bin_centers = _frequency_bin_centers(n, device=device)
    df = torch.atleast_1d(bin_centers[1])
    bin_centers = torch.concatenate([bin_centers, 0.5 + df], dim=0)  # (b+1, )
    adjacent_bins = bin_centers.unfold(dimension=0, size=2, step=1)  # (b, 2)
    return einops.reduce(adjacent_bins, "b high_low -> b", reduction="mean")


def rfft_shape(
    input_shape: Sequence[int],
) -> tuple[int, ...]:
    """
    Get the output shape of an rfft on an input with input_shape.

    Parameters
    ----------
    input_shape : Sequence[int]
        The shape of the input tensor

    Returns
    -------
    Tuple[int]
        The shape of the output tensor
    """
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)
