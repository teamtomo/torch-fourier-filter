"""Dose weighting functions for Fourier filtering."""

import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid


def critical_exposure(fft_freq: torch.Tensor) -> torch.Tensor:
    """
    Calculate the critical exposure using the Grant and Grigorieff 2015 formula.

    Args:
        fft_freq: The frequency grid of the Fourier transform.

    Returns
    -------
        The critical exposure for the given frequency grid
    """
    a = 0.245
    b = -1.665
    c = 2.81

    eps = 1e-10
    Ne = a * torch.pow(fft_freq.clamp(min=eps), b) + c
    return Ne


def critical_exposure_bfactor(fft_freq: torch.Tensor, bfactor: float) -> torch.Tensor:
    """
    Calculate the critical exposure using a user defined B-factor.

    Args:
        fft_freq: The frequency grid of the Fourier transform.
        bfactor: The B-factor to use.

    Returns
    -------
        The critical exposure for the given frequency grid
    """
    eps = 1e-10
    Ne = 4 / (bfactor * fft_freq.clamp(min=eps) ** 2)
    return Ne


def cumulative_dose_filter_3d(
    volume_shape: tuple[int, int, int] | tuple[int, int],
    pixel_size: float = 1,
    start_exposure: float = 0.0,
    end_exposure: float = 30.0,
    crit_exposure_bfactor: int | float = -1,
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Dose weight a 3D volume using Grant and Grigorieff 2015.

    Use integration to speed up.

    Parameters
    ----------
    volume_shape : tuple[int, int, int]
        The shape of the filter to calculate (real space). Rfft is
        automatically calculated from this.
    pixel_size : float
        The pixel size of the volume, in Angstrom.
    start_exposure : float
        The start exposure for dose weighting, in e-/A^2. Default is 0.0.
    end_exposure : float
        The end exposure for dose weighting, in e-/A^2. Default is 30.0.
    crit_exposure_bfactor : int | float
        The B factor for dose weighting based on critical exposure. If '-1',
        then use Grant and Grigorieff (2015) values.
    rfft : bool
        If the FFT is a real FFT.
    fftshift : bool
        If the FFT is shifted.
    device : torch.device
        The device to use for the calculation.

    Returns
    -------
    torch.Tensor
        The dose weighting filter.
    """
    # Get the frequency grid for 1 frame
    fft_freq_px = fftfreq_grid(
        image_shape=volume_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=device,
    )
    fft_freq_px /= pixel_size  # Convert to Angstrom^-1

    # Get the critical exposure for each frequency
    if crit_exposure_bfactor == -1:
        Ne = critical_exposure(fft_freq=fft_freq_px)
    elif crit_exposure_bfactor >= 0:
        Ne = critical_exposure_bfactor(
            fft_freq=fft_freq_px, bfactor=crit_exposure_bfactor
        )
    else:
        raise ValueError("B-factor must be positive or -1.")

    # Add small epsilon to prevent division by zero
    eps = 1e-10
    Ne = Ne.clamp(min=eps)

    return (
        2
        * Ne
        * (
            torch.exp((-0.5 * start_exposure) / Ne)
            - torch.exp((-0.5 * end_exposure) / Ne)
        )
        / end_exposure
    )
