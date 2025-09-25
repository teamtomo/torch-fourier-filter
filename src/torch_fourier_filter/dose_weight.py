"""Dose weighting functions for Fourier filtering."""

import einops
import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid


def critical_exposure(
    fft_freq: torch.Tensor,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
) -> torch.Tensor:
    """
    Calculate the critical exposure using the Grant and Grigorieff 2015 formula.

    Ne = a * fft_freq^b + c

    Parameters
    ----------
    fft_freq: torch.Tensor
        The frequency grid of the Fourier transform.
    a: float
        The a parameter for the critical exposure formula. Default is 0.245.
    b: float
        The b parameter for the critical exposure formula. Default is -1.665.
    c: float
        The c parameter for the critical exposure formula. Default is 2.81.

    Returns
    -------
        The critical exposure for the given frequency grid.
    """
    eps = 1e-10
    Ne = a * torch.pow(fft_freq.clamp(min=eps), b) + c
    return Ne


def critical_exposure_bfactor(fft_freq: torch.Tensor, bfactor: float) -> torch.Tensor:
    """
    Calculate the critical exposure using a user defined B-factor.

    Parameters
    ----------
    fft_freq: torch.Tensor
        The frequency grid of the Fourier transform.
    bfactor: float
        The B-factor to use.

    Returns
    -------
        The critical exposure for the given frequency grid.
    """
    eps = 1e-10
    Ne = 4 / (bfactor * fft_freq.clamp(min=eps) ** 2)
    return Ne


def dose_weight_2d(
    image_dft: torch.Tensor,  # shape (..., h, w)
    image_shape: tuple[int, int],  # shape (h, w)
    pixel_size: float,
    dose: torch.Tensor | float,  # shape (..., ) or float
    voltage: float = 300.0,
    crit_exposure_bfactor: int | float = -1,
    rfft: bool = True,
    fftshift: bool = False,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Apply dose weighting to an image or stack.

    This function implements the dose weighting algorithm following Grant and
    Grigorieff 2015, applying different weights to each frame based on cumulative
    dose and then normalizing across frames.

    Parameters
    ----------
    image_dft : torch.Tensor
        Complex tensor containing imagesin Fourier space with shape
        (..., h, w) for rfft=True or (..., h, w) for full fft.
    image_shape : tuple[int, int]
        The shape of the real space images (h, w).
    pixel_size : float
        The pixel size of the images, in Angstroms.
    dose : torch.Tensor | float, optional
        The dose, in e-/A^2.
    voltage : float, optional
        The acceleration voltage in kV. Affects damage correction for 100kV and 200kV.
        Default is 300.0.
    crit_exposure_bfactor : int | float, optional
        The B factor for dose weighting based on critical exposure. If -1,
        then use Grant and Grigorieff (2015) values. Default is -1.
    rfft : bool, optional
        Whether the input DFT is from a real FFT. Default is True.
    fftshift : bool, optional
        Whether the input DFT is fftshifted. Default is False.
    a : float, optional
        The a parameter for the critical exposure formula. Default is 0.245.
    b : float, optional
        The b parameter for the critical exposure formula. Default is -1.665.
    c : float, optional
        The c parameter for the critical exposure formula. Default is 2.81.
    device : torch.device | None, optional
        The device to use for the calculation. If None, infers device from movie_dft.
        Default is None.

    Returns
    -------
    torch.Tensor
        The dose-weighted movie frames with the same shape as input.
    """
    # If dose is a tensor with more than 1 value,
    # check it matches the ... batch dimensions in the image.
    if isinstance(dose, torch.Tensor) and dose.numel() > 1 and image_dft.ndim > 2:
        # image_dft shape: (..., h, w)
        image_batch_shape = image_dft.shape[:-2]
        dose_shape = dose.shape
        if dose_shape != image_batch_shape:
            raise ValueError(
                f"dose tensor shape {dose_shape} "
                f"does not match image batch dimensions {image_batch_shape}"
            )
        else:
            dose = einops.rearrange(dose, "... -> ... 1 1")

    # Determine device
    if device is None:
        device = image_dft.device

    # Move movie_dft to specified device if needed
    image_dft = image_dft.to(device)

    # Apply voltage-dependent damage corrections (follows RELION)
    if abs(voltage - 200) <= 2:
        dose /= 0.8
    elif abs(voltage - 100) <= 2:
        dose /= 0.64

    # Get frequency grid
    fft_freq_px = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=device,
    )
    fft_freq_px /= pixel_size  # Convert to Angstrom^-1

    # Calculate critical exposure for each frequency
    if crit_exposure_bfactor == -1:
        Ne = critical_exposure(fft_freq=fft_freq_px, a=a, b=b, c=c)
    elif crit_exposure_bfactor >= 0:
        Ne = critical_exposure_bfactor(
            fft_freq=fft_freq_px, bfactor=crit_exposure_bfactor
        )
    else:
        raise ValueError("B-factor must be positive or -1.")

    # Apply factor of 2 from Eq. 5 (factoring out 0.5)
    Ne = Ne * 2

    # Add small epsilon to prevent division by zero
    eps = 1e-10
    Ne = Ne.clamp(min=eps)

    # Calculate weights for each frame at each frequency
    # Reshape for broadcasting: dose (..., 1, 1) and Ne (1, h, w)
    # Expand Ne to match the number of leading dimensions in dose/image_dft
    n_leading = len(image_dft.shape) - 2  # number of ... dims
    Ne_expanded = einops.rearrange(Ne, "h w -> " + " ".join(["1"] * n_leading) + " h w")
    weights = torch.exp(-dose / Ne_expanded)  # Shape: (..., h, w)

    # Apply weights to each image
    weighted_images = image_dft * weights

    # Calculate sum of squared weights for normalization (Eq. 9)
    sum_weight_sq = einops.reduce(weights**2, "... h w -> h w", "sum")
    sum_weight_sq = torch.sqrt(sum_weight_sq.clamp(min=eps))

    # Normalize all frames by the sum of squared weights
    sum_weight_sq_expanded = einops.rearrange(
        sum_weight_sq, "h w -> " + " ".join(["1"] * n_leading) + " h w"
    )
    normalized_frames = weighted_images / sum_weight_sq_expanded

    return normalized_frames


def dose_weight_movie(
    movie_dft: torch.Tensor,
    image_shape: tuple[int, int],
    pixel_size: float,
    pre_exposure: float = 0.0,
    dose_per_frame: float = 1.0,
    voltage: float = 300.0,
    crit_exposure_bfactor: int | float = -1,
    rfft: bool = True,
    fftshift: bool = False,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Apply per-frame dose weighting to a movie in Fourier space.

    This function implements the dose weighting algorithm following Grant and
    Grigorieff 2015, applying different weights to each frame based on cumulative
    dose and then normalizing across frames.

    Parameters
    ----------
    movie_dft : torch.Tensor
        Complex tensor containing movie frames in Fourier space with shape
        (n_frames, h, w) for rfft=True or (n_frames, h, w) for full fft.
    image_shape : tuple[int, int]
        The shape of the real space images (h, w).
    pixel_size : float
        The pixel size of the images, in Angstroms.
    pre_exposure : float, optional
        The pre-exposure before the first frame, in e-/A^2. Default is 0.0.
    dose_per_frame : float, optional
        The dose per frame, in e-/A^2. Default is 1.0.
    voltage : float, optional
        The acceleration voltage in kV. Affects damage correction for 100kV and 200kV.
        Default is 300.0.
    crit_exposure_bfactor : int | float, optional
        The B factor for dose weighting based on critical exposure. If -1,
        then use Grant and Grigorieff (2015) values. Default is -1.
    rfft : bool, optional
        Whether the input DFT is from a real FFT. Default is True.
    fftshift : bool, optional
        Whether the input DFT is fftshifted. Default is False.
    a : float, optional
        The a parameter for the critical exposure formula. Default is 0.245.
    b : float, optional
        The b parameter for the critical exposure formula. Default is -1.665.
    c : float, optional
        The c parameter for the critical exposure formula. Default is 2.81.
    device : torch.device | None, optional
        The device to use for the calculation. If None, infers device from movie_dft.
        Default is None.

    Returns
    -------
    torch.Tensor
        The dose-weighted movie frames with the same shape as input.
    """
    if movie_dft.ndim != 3:
        raise ValueError(
            f"movie_dft must be 3D tensor with shape (n_frames, h, w),"
            f" got {movie_dft.shape}"
        )

    # Determine device
    if device is None:
        device = movie_dft.device

    # Move movie_dft to specified device if needed
    movie_dft = movie_dft.to(device)
    # Calculate doses for each frame (dose AFTER each frame)
    frame_indices = torch.arange(movie_dft.shape[0], dtype=torch.float32, device=device)
    doses = pre_exposure + dose_per_frame * (frame_indices + 1)

    normalized_frames = dose_weight_2d(
        image_dft=movie_dft,
        image_shape=image_shape,
        pixel_size=pixel_size,
        dose=doses,
        voltage=voltage,
        crit_exposure_bfactor=crit_exposure_bfactor,
        rfft=rfft,
        fftshift=fftshift,
        a=a,
        b=b,
        c=c,
        device=device,
    )

    return normalized_frames


def cumulative_dose_filter_3d(
    volume_shape: tuple[int, int, int] | tuple[int, int],
    pixel_size: float = 1,
    start_exposure: float = 0.0,
    end_exposure: float = 30.0,
    crit_exposure_bfactor: int | float = -1,
    rfft: bool = True,
    fftshift: bool = False,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
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
    a: float
        The a parameter for the critical exposure formula. Default is 0.245.
    b: float
        The b parameter for the critical exposure formula. Default is -1.665.
    c: float
        The c parameter for the critical exposure formula. Default is 2.81.
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
        Ne = critical_exposure(fft_freq=fft_freq_px, a=a, b=b, c=c)
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
