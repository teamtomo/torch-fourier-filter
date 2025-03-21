"""CTF module for Fourier filtering."""

import einops as _einops
import torch as _torch
from scipy import constants as _C
from torch_grid_utils.fftfreq_grid import fftfreq_grid as _fftfreq_grid


def calculate_relativistic_electron_wavelength(energy: float | _torch.Tensor ) -> _torch.Tensor:
    """Calculate the relativistic electron wavelength in SI units.

    For derivation see:
    1.  Kirkland, E. J. Advanced Computing in Electron Microscopy.
        (Springer International Publishing, 2020). doi:10.1007/978-3-030-33260-0.

    2.  https://en.wikipedia.org/wiki/Electron_diffraction#Relativistic_theory

    Parameters
    ----------
    energy: float | torch.Tensor
        acceleration potential in volts.

    Returns
    -------
    wavelength: float | torch.Tensor
        relativistic wavelength of the electron in meters.
    """
    h = __C.Planck
    c = _C.speed_of_light
    m0 = _C.electron_mass
    e = _C.elementary_charge
    V = _torch.as_tensor(energy, dtype=_torch.float)
    eV = e * V

    numerator = h * c
    denominator = _torch.sqrt(eV * (2 * m0 * c**2 + eV))
    return numerator / denominator


def calculate_ctf_2d(
    defocus: float | _torch.Tensor,
    astigmatism: float | _torch.Tensor,
    astigmatism_angle: float | _torch.Tensor,
    voltage: float | _torch.Tensor,
    spherical_aberration: float | _torch.Tensor,
    amplitude_contrast: float | _torch.Tensor,
    b_factor: float | _torch.Tensor,
    phase_shift: float | _torch.Tensor,
    pixel_size: float | _torch.Tensor,
    image_shape: tuple[int, int],
    rfft: bool,
    fftshift: bool,
) -> _torch.Tensor:
    """Calculate the Contrast Transfer Function (CTF) for a 2D image.

    NOTE: The device of the input tensors is inferred from the `defocus` tensor.

    Parameters
    ----------
    defocus: float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
        `(defocus_u + defocus_v) / 2`
    astigmatism: float | torch.Tensor
        Amount of astigmatism in micrometers.
        `(defocus_u - defocus_v) / 2`
    astigmatism_angle: float | torch.Tensor
        Angle of astigmatism in degrees. 0 places `defocus_u` along the y-axis.
    pixel_size: float | torch.Tensor
        Pixel size in Angströms per pixel (Å px⁻¹).
    voltage: float | torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration: float | torch.Tensor
        Spherical aberration in millimeters (mm).
    amplitude_contrast: float | torch.Tensor
        Fraction of amplitude contrast (value in range [0, 1]).
    b_factor: float | torch.Tensor
        B-factor in square angstroms. Should be positive
    phase_shift: float | torch.Tensor
        Angle of phase shift applied to CTF in degrees.
    image_shape: Tuple[int, int]
        Shape of 2D images onto which CTF will be applied.
    rfft: bool
        Generate the CTF containing only the non-redundant half transform from a rfft.
    fftshift: bool
        Whether to apply fftshift on the resulting CTF images.

    Returns
    -------
    ctf: _torch.Tensor
        The Contrast Transfer Function for the given parameters.
    """
    if isinstance(defocus, _torch.Tensor):
        device = defocus.device
    else:
        device = _torch.device("cpu")

    # to _torch.Tensor
    defocus = _torch.as_tensor(defocus, dtype=_torch.float, device=device)
    astigmatism = _torch.as_tensor(astigmatism, dtype=_torch.float, device=device)
    astigmatism_angle = _torch.as_tensor(astigmatism_angle, dtype=_torch.float, device=device)
    pixel_size = _torch.as_tensor(pixel_size, dtype=_torch.float, device=device)
    voltage = _torch.as_tensor(voltage, dtype=_torch.float, device=device)
    spherical_aberration = _torch.as_tensor(spherical_aberration, dtype=_torch.float, device=device)
    amplitude_contrast = _torch.as_tensor(amplitude_contrast, dtype=_torch.float, device=device)
    b_factor = _torch.as_tensor(b_factor, dtype=_torch.float, device=device)
    phase_shift = _torch.as_tensor(phase_shift, dtype=_torch.float, device=device)
    image_shape = _torch.as_tensor(image_shape, dtype=_torch.int, device=device)

    # Unit conversions
    defocus *= 1e4  # micrometers -> angstroms
    astigmatism *= 1e4  # micrometers -> angstroms
    astigmatism_angle *= _C.pi / 180  # degrees -> radians
    voltage *= 1e3  # kV -> V
    spherical_aberration *= 1e7  # mm -> angstroms

    # derived quantities used in CTF calculation
    defocus_u = defocus + astigmatism
    defocus_v = defocus - astigmatism
    _lambda = (
        calculate_relativistic_electron_wavelength(voltage) * 1e10
    ) # meters -> angstroms
    k1 = -_C.pi * _lambda
    k2 = _C.pi / 2 * spherical_aberration * _lambda**3
    k3 = _torch.deg2rad(_torch.as_tensor(phase_shift, dtype=_torch.float))
    k4 = -b_factor / 4
    k5 = _torch.arctan(amplitude_contrast / _torch.sqrt(1 - amplitude_contrast**2))
    
    k1 = _einops.rearrange(k1, "... -> ... 1 1")
    k2 = _einops.rearrange(k2, "... -> ... 1 1")
    k3 = _einops.rearrange(k3, "... -> ... 1 1")
    k4 = _einops.rearrange(k4, "... -> ... 1 1")
    k5 = _einops.rearrange(k5, "... -> ... 1 1")
    
    # construct 2D frequency grids and rescale cycles / px -> cycles / Å
    fft_freq_grid = _fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=False,
        device=device,
    )
    fft_freq_grid = fft_freq_grid / _einops.rearrange(pixel_size, "... -> ... 1 1 1")
    _fftfreq_grid_squared = fft_freq_grid**2

    # Astigmatism
    #         Q = [[ sin, cos]
    #              [-sin, cos]]
    #         D = [[   u,   0]
    #              [   0,   v]]
    #         A = Q^T.D.Q = [[ Axx, Axy]
    #                        [ Ayx, Ayy]]
    #         Axx = cos^2 * u + sin^2 * v
    #         Ayy = sin^2 * u + cos^2 * v
    #         Axy = Ayx = cos * sin * (u - v)
    #         defocus = A.k.k^2 = Axx*x^2 + 2*Axy*x*y + Ayy*y^2

    c = _torch.cos(astigmatism_angle)
    c2 = c**2
    s = _torch.sin(astigmatism_angle)
    s2 = s**2
    
    yy2, xx2 = _einops.rearrange(_fftfreq_grid_squared, "... h w freq -> freq ... h w")
    xy = _einops.reduce(fft_freq_grid, "... h w freq -> ... h w", reduction="prod")
    n4 = (
        _einops.reduce(_fftfreq_grid_squared, "... h w freq -> ... h w", reduction="sum") ** 2
    )
    
    Axx = c2 * defocus_u + s2 * defocus_v
    Axy = c * s * (defocus_u - defocus_v)
    Ayy = s2 * defocus_u + c2 * defocus_v
    
    Ayy_y2 = _einops.rearrange(Ayy, "... -> ... 1 1") * yy2
    Axx_x2 = _einops.rearrange(Axx, "... -> ... 1 1") * xx2
    Axy_xy = _einops.rearrange(Axy, "... -> ... 1 1") * xy
    
    # calculate ctf
    ctf = -_torch.sin(k1 * (Axx_x2 + (2 * Axy_xy) + Ayy_y2) + k2 * n4 - k3 - k5)

    if _torch.any(k4 < 0):
        idx = (k4 < 0).squeeze(dim=(-2,-1)) 
        ctf[idx] *= _torch.exp(k4[idx] * n4[idx])
    return ctf


def calculate_ctf_1d(
    defocus: float | _torch.Tensor,
    voltage: float | _torch.Tensor,
    spherical_aberration: float | _torch.Tensor,
    amplitude_contrast: float | _torch.Tensor,
    b_factor: float | _torch.Tensor,
    phase_shift: float | _torch.Tensor,
    pixel_size: float | _torch.Tensor,
    n_samples: int,
    oversampling_factor: int,
) -> _torch.Tensor:
    """
    Calculate the Contrast Transfer Function (CTF) for a 1D signal.

    Parameters
    ----------
    defocus: float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
    pixel_size: float | torch.Tensor
        Pixel size in Angströms per pixel (Å px⁻¹).
    voltage: float | torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration: float | torch.Tensor
        Spherical aberration in millimeters (mm).
    amplitude_contrast: float | torch.Tensor
        Fraction of amplitude contrast (value in range [0, 1]).
    b_factor: float | torch.Tensor
        B-factor in square angstroms. Should be positive
    phase_shift: float | torch.Tensor
        Angle of phase shift applied to CTF in degrees.
    n_samples: int
        Number of samples in CTF.
    oversampling_factor: int
        Factor by which to oversample the CTF.

    Returns
    -------
    ctf: torch.Tensor
        The Contrast Transfer Function for the given parameters
    """
    if isinstance(defocus, _torch.Tensor):
        device = defocus.device
    else:
        device = _torch.device("cpu")
        
    # to _torch.Tensor
    defocus = _torch.as_tensor(defocus, dtype=_torch.float, device = device)
    pixel_size = _torch.as_tensor(pixel_size, dtype=_torch.int, device = device)
    voltage = _torch.as_tensor(voltage, dtype=_torch.float, device = device)
    spherical_aberration = _torch.as_tensor(spherical_aberration, dtype=_torch.float, device = device)
    amplitude_contrast = _torch.as_tensor(amplitude_contrast, dtype=_torch.float, device = device)
    b_factor = _torch.as_tensor(b_factor, dtype=_torch.float, device = device)
    phase_shift = _torch.as_tensor(phase_shift, dtype=_torch.float, device = device)
    
    # Unit conversions
    defocus = defocus * 1e4  # micrometers -> angstroms
    voltage = voltage * 1e3  # kV -> V
    spherical_aberration *= 1e7  # mm -> angstroms
    
    # derived quantities used in CTF calculation
    _lambda = (
        calculate_relativistic_electron_wavelength(voltage) * 1e10
    )  # meters -> angstroms
    
    k1 = -_C.pi * _lambda
    k2 = _C.pi / 2 * spherical_aberration * _lambda**3
    k3 = _torch.deg2rad(_torch.as_tensor(phase_shift, dtype=_torch.float))
    k4 = -b_factor / 4
    k5 = _torch.arctan(amplitude_contrast / _torch.sqrt(1 - amplitude_contrast**2))
    
    k1 = _einops.rearrange(k1, "... -> ... 1 1")
    k2 = _einops.rearrange(k2, "... -> ... 1 1")
    k3 = _einops.rearrange(k3, "... -> ... 1 1")
    k4 = _einops.rearrange(k4, "... -> ... 1 1")
    k5 = _einops.rearrange(k5, "... -> ... 1 1")
    defocus = _einops.rearrange(defocus, "... -> ... 1 1") # oversampling dim
    pixel_size = _einops.rearrange(pixel_size, "... -> ... 1 1") # oversampling dim
    
    # construct frequency vector and rescale cycles / px -> cycles / Å
    _fftfreq_grid = _torch.linspace(0, 0.5, steps=n_samples)  # (n_samples, 
    
    # oversampling...
    if oversampling_factor > 1:
        frequency_delta = 0.5 / (n_samples - 1)
        oversampled_frequency_delta = frequency_delta / oversampling_factor
        oversampled_interval_length = oversampled_frequency_delta * (
            oversampling_factor - 1
        )
        per_frequency_deltas = _torch.linspace(
            0, oversampled_interval_length, steps=oversampling_factor
        )
        per_frequency_deltas -= oversampled_interval_length / 2
        per_frequency_deltas = _einops.rearrange(per_frequency_deltas, "os -> os 1")
        _fftfreq_grid = _fftfreq_grid + per_frequency_deltas
    _fftfreq_grid = _fftfreq_grid / pixel_size
    _fftfreq_grid_squared = _fftfreq_grid**2
    n4 = _fftfreq_grid_squared**2
    
    # calculate ctf
    ctf = -_torch.sin(k1 * _fftfreq_grid_squared * defocus + k2 * n4 - k3 - k5)
    if _torch.any(k4 < 0):
        idx = (k4 < 0).squeeze(dim=(-2,-1)) 
        ctf[idx] *= _torch.exp(k4[idx] * n4[idx])
    ctf = _einops.reduce(ctf, "... os k -> ... k", reduction="mean") # oversampling reduction
    return ctf