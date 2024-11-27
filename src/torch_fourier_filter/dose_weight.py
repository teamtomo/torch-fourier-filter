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


def critical_exposure_Bfac(fft_freq: torch.Tensor, Bfac: float) -> torch.Tensor:
    """
    Calculate the critical exposure using a user defined B-factor.

    Args:
        fft_freq: The frequency grid of the Fourier transform.
        Bfac: The B-factor to use.

    Returns
    -------
        The critical exposure for the given frequency grid
    """
    eps = 1e-10
    Ne = 2 / (Bfac * fft_freq.clamp(min=eps) ** 2)
    return Ne


def cumulative_dose_filter_3d(
    volume_shape: tuple[int, int, int],
    num_frames: int,
    start_exposure: float = 0.0,
    pixel_size: float = 1,
    flux: float = 1,
    Bfac: float = -1,
    rfft: bool = True,
    fftshift: bool = False,
) -> torch.Tensor:
    """
    Dose weight a 3D volume using Grant and Grigorieff 2015.

    Use integration to speed up.

    Parameters
    ----------
    volume_shape : tuple[int, int, int]
        The volume shape for dose weighting.
    num_frames : int
        The number of frames for dose weighting.
    start_exposure : float
        The start exposure for dose weighting.
    pixel_size : float
        The pixel size of the volume.
    flux : float
        The fluence per frame.
    Bfac : float
        The B factor for dose weighting, -1=use Grant and Grigorieff values.
    rfft : bool
        If the FFT is a real FFT.
    fftshift : bool
        If the FFT is shifted.

    Returns
    -------
    torch.Tensor
        The dose weighting filter.
    """
    end_exposure = start_exposure + num_frames * flux
    # Get the frequency grid for 1 frame
    fft_freq_px = (
        fftfreq_grid(
            image_shape=volume_shape,
            rfft=rfft,
            fftshift=fftshift,
            norm=True,
        )
        / pixel_size
    )

    # Get the critical exposure for each frequency
    Ne = (
        critical_exposure_Bfac(fft_freq=fft_freq_px, Bfac=Bfac)
        if Bfac >= 0
        else critical_exposure(fft_freq=fft_freq_px)
    )

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
