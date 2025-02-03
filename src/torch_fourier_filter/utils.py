"""Utility functions related to calculating and constructing Fourier filters."""

from typing import Optional

import torch


def torch_interp(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    left: Optional[float | complex] = None,
    right: Optional[float | complex] = None,
    period: Optional[float] = None,
) -> torch.Tensor:
    """Reimplementation of np.interp in PyTorch for 1d linear interpolation.

    NOTE: Period function is not implemented.

    Parameters
    ----------
    x : torch.Tensor
        The x-coordinates at which to evaluate the interpolated values.
    xp : torch.Tensor
        The x-coordinates of the data points.
    fp : torch.Tensor
        The y-coordinates of the data points, must be same length as xp.
    left : float, optional
        Value to return for x < xp[0], default is fp[0].
    right : float, optional
        Value to return for x > xp[-1], default is fp[-1].
    period : float, optional
        Not implemented!

    Returns
    -------
    torch.Tensor
        The interpolated values, same shape as x.

    Raises
    ------
    NotImplementedError
        If period is not None.
    """
    if period is not None:
        raise NotImplementedError("Periodic interpolation is not implemented.")

    if xp.ndim != 1 or fp.ndim != 1:
        raise ValueError("xp and fp must be both 1D tensors.")
    if xp.shape != fp.shape:
        raise ValueError("xp and fp must have the same shape.")

    # left and right fill values
    if left is None:
        left = fp[0]
    if right is None:
        right = fp[-1]

    # find the indices of the left and right values
    i = torch.searchsorted(xp, x, side="right")

    # Out of bounds masks
    oob_left = x < xp[0]
    oob_right = x > xp[-1]

    # index-in-bounds (iib) and x-in-bounds (xib)
    iib = i[(~oob_left) & (~oob_right)]
    xib = x[(~oob_left) & (~oob_right)]

    f_lo = fp[iib - 1]
    f_hi = fp[iib]
    x_lo = xp[iib - 1]
    x_hi = xp[iib]

    # Construct output tensor and do liner interpolation
    out = torch.empty_like(x)
    out[(~oob_left) & (~oob_right)] = f_lo + (f_hi - f_lo) * (xib - x_lo) / (
        x_hi - x_lo
    )
    out[oob_left] = left
    out[oob_right] = right

    return out


def curve_1dim_to_ndim(
    values_1d: torch.Tensor,  # shape (batch, N) or (N,)
    frequency_1d: torch.Tensor,  # shape (N,)
    frequency_grid: torch.Tensor,
    fill_lower: Optional[float | complex] = None,
    fill_upper: Optional[float | complex] = None,
) -> torch.Tensor:
    """Use interpolation to map 1D values from a curve to a nD grid.

    Parameters
    ----------
    values_1d : torch.Tensor
        Values of the one-dimensional curve corresponding to each frequency value.
    frequency_1d : torch.Tensor
        Corresponding frequency values of the one-dimensional curve.
    frequency_grid : torch.Tensor
        Absolute frequency values of the grid.
    fill_lower : float | complex, optional
        Fill value for lower (left) out-of-bound frequencies. Default is None and
        corresponds to the lowest (leftmost) value in values_1d.
    fill_upper : float | complex, optional
        Fill value for upper (right) out-of-bound frequencies. Default is None and
        corresponds to the highest (rightmost) value in values_1d.

    Returns
    -------
    torch.Tensor
        nD filter constructed from the 1D values.

    Example
    -------
    >>> values_1d = torch.tensor([1.0, 2.0, 7.0])
    >>> frequency_1d = torch.tensor([0.0, 0.25, 0.5])
    >>> grid = torch.tensor([[0.0. 0.1, 0.2], [0.3, 0.4, 0.5]])  # 2D toy example
    >>> filter_2d = curve_1dim_to_ndim(values_1d, frequency_1d, grid)
    >>> filter_2d
    tensor([[1.0, 1.5, 2.0],
            [4.0, 5.5, 7.0]])
    """
    if values_1d.ndim > 2:
        raise ValueError("values_1d must be at most 2D tensor.")

    if values_1d.ndim == 1:
        values_1d = values_1d.unsqueeze(0)

    interpn_values = torch.empty(values_1d.shape[0], *frequency_grid.shape)

    for i in range(values_1d.shape[0]):
        interpn_values[i] = torch_interp(
            frequency_grid.ravel(),
            frequency_1d,
            values_1d[i],
            left=fill_lower,
            right=fill_upper,
        ).reshape(frequency_grid.shape)

    return interpn_values
