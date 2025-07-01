"""CTF module for Fourier filtering."""

import einops
import torch
from scipy import constants as C
from torch_grid_utils.fftfreq_grid import fftfreq_grid


def calculate_relativistic_electron_wavelength(energy: float | torch.Tensor ) -> torch.Tensor:
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
    h = C.Planck
    c = C.speed_of_light
    m0 = C.electron_mass
    e = C.elementary_charge
    V = torch.as_tensor(energy, dtype=torch.float)
    eV = e * V

    numerator = h * c
    denominator = torch.sqrt(eV * (2 * m0 * c**2 + eV))
    return numerator / denominator


def calculate_ctf_2d(
    defocus: float | torch.Tensor,
    astigmatism: float | torch.Tensor,
    astigmatism_angle: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    b_factor: float | torch.Tensor,
    phase_shift: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    image_shape: tuple[int, int],
    rfft: bool,
    fftshift: bool,
) -> torch.Tensor:
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
    ctf: torch.Tensor
        The Contrast Transfer Function for the given parameters.
    """
    if isinstance(defocus, torch.Tensor):
        device = defocus.device
    else:
        device = torch.device("cpu")

    # to torch.Tensor
    defocus = torch.as_tensor(defocus, dtype=torch.float, device=device)
    astigmatism = torch.as_tensor(astigmatism, dtype=torch.float, device=device)
    astigmatism_angle = torch.as_tensor(astigmatism_angle, dtype=torch.float, device=device)
    pixel_size = torch.as_tensor(pixel_size, dtype=torch.float, device=device)
    voltage = torch.as_tensor(voltage, dtype=torch.float, device=device)
    spherical_aberration = torch.as_tensor(spherical_aberration, dtype=torch.float, device=device)
    amplitude_contrast = torch.as_tensor(amplitude_contrast, dtype=torch.float, device=device)
    b_factor = torch.as_tensor(b_factor, dtype=torch.float, device=device)
    phase_shift = torch.as_tensor(phase_shift, dtype=torch.float, device=device)
    image_shape = torch.as_tensor(image_shape, dtype=torch.int, device=device)

    # Unit conversions
    defocus *= 1e4  # micrometers -> angstroms
    astigmatism *= 1e4  # micrometers -> angstroms
    astigmatism_angle *= C.pi / 180  # degrees -> radians
    voltage *= 1e3  # kV -> V
    spherical_aberration *= 1e7  # mm -> angstroms

    # derived quantities used in CTF calculation
    defocus_u = defocus + astigmatism
    defocus_v = defocus - astigmatism
    _lambda = (
        calculate_relativistic_electron_wavelength(voltage) * 1e10
    ) # meters -> angstroms
    k1 = -C.pi * _lambda
    k2 = C.pi / 2 * spherical_aberration * _lambda**3
    k3 = torch.deg2rad(torch.as_tensor(phase_shift, dtype=torch.float))
    k4 = -b_factor / 4
    k5 = torch.arctan(amplitude_contrast / torch.sqrt(1 - amplitude_contrast**2))
    
    k1 = einops.rearrange(k1, "... -> ... 1 1")
    k2 = einops.rearrange(k2, "... -> ... 1 1")
    k3 = einops.rearrange(k3, "... -> ... 1 1")
    k4 = einops.rearrange(k4, "... -> ... 1 1")
    k5 = einops.rearrange(k5, "... -> ... 1 1")
    
    # construct 2D frequency grids and rescale cycles / px -> cycles / Å
    fft_freq_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=False,
        device=device,
    )
    fft_freq_grid = fft_freq_grid / einops.rearrange(pixel_size, "... -> ... 1 1 1")
    fftfreq_grid_squared = fft_freq_grid**2

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

    c = torch.cos(astigmatism_angle)
    c2 = c**2
    s = torch.sin(astigmatism_angle)
    s2 = s**2
    
    yy2, xx2 = einops.rearrange(fftfreq_grid_squared, "... h w freq -> freq ... h w")
    xy = einops.reduce(fft_freq_grid, "... h w freq -> ... h w", reduction="prod")
    n4 = (
        einops.reduce(fftfreq_grid_squared, "... h w freq -> ... h w", reduction="sum") ** 2
    )
    
    Axx = c2 * defocus_u + s2 * defocus_v
    Axy = c * s * (defocus_u - defocus_v)
    Ayy = s2 * defocus_u + c2 * defocus_v
    
    Ayy_y2 = einops.rearrange(Ayy, "... -> ... 1 1") * yy2
    Axx_x2 = einops.rearrange(Axx, "... -> ... 1 1") * xx2
    Axy_xy = einops.rearrange(Axy, "... -> ... 1 1") * xy
    
    # calculate ctf
    ctf = -torch.sin(k1 * (Axx_x2 + (2 * Axy_xy) + Ayy_y2) + k2 * n4 - k3 - k5)

    if torch.any(k4 < 0):
        idx = (k4 < 0).squeeze(dim=(-2,-1)) 
        ctf[idx] *= torch.exp(k4[idx] * n4[idx])
    return ctf


def calculate_ctf_1d(
    defocus: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    b_factor: float | torch.Tensor,
    phase_shift: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    n_samples: int,
    oversampling_factor: int,
) -> torch.Tensor:
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
    if isinstance(defocus, torch.Tensor):
        device = defocus.device
    else:
        device = torch.device("cpu")
        
    # to torch.Tensor
    defocus = torch.as_tensor(defocus, dtype=torch.float, device = device)
    pixel_size = torch.as_tensor(pixel_size, dtype=torch.float, device = device)
    voltage = torch.as_tensor(voltage, dtype=torch.float, device = device)
    spherical_aberration = torch.as_tensor(spherical_aberration, dtype=torch.float, device = device)
    amplitude_contrast = torch.as_tensor(amplitude_contrast, dtype=torch.float, device = device)
    b_factor = torch.as_tensor(b_factor, dtype=torch.float, device = device)
    phase_shift = torch.as_tensor(phase_shift, dtype=torch.float, device = device)
    
    # Unit conversions
    defocus = defocus * 1e4  # micrometers -> angstroms
    voltage = voltage * 1e3  # kV -> V
    spherical_aberration *= 1e7  # mm -> angstroms
    
    # derived quantities used in CTF calculation
    _lambda = (
        calculate_relativistic_electron_wavelength(voltage) * 1e10
    )  # meters -> angstroms
    
    k1 = -C.pi * _lambda
    k2 = C.pi / 2 * spherical_aberration * _lambda**3
    k3 = torch.deg2rad(torch.as_tensor(phase_shift, dtype=torch.float))
    k4 = -b_factor / 4
    k5 = torch.arctan(amplitude_contrast / torch.sqrt(1 - amplitude_contrast**2))
    
    k1 = einops.rearrange(k1, "... -> ... 1 1")
    k2 = einops.rearrange(k2, "... -> ... 1 1")
    k3 = einops.rearrange(k3, "... -> ... 1 1")
    k4 = einops.rearrange(k4, "... -> ... 1 1")
    k5 = einops.rearrange(k5, "... -> ... 1 1")
    defocus = einops.rearrange(defocus, "... -> ... 1 1") # oversampling dim
    pixel_size = einops.rearrange(pixel_size, "... -> ... 1 1") # oversampling dim
    
    # construct frequency vector and rescale cycles / px -> cycles / Å
    fftfreq_grid = torch.linspace(0, 0.5, steps=n_samples)  # (n_samples, 
    
    # oversampling...
    if oversampling_factor > 1:
        frequency_delta = 0.5 / (n_samples - 1)
        oversampled_frequency_delta = frequency_delta / oversampling_factor
        oversampled_interval_length = oversampled_frequency_delta * (
            oversampling_factor - 1
        )
        per_frequency_deltas = torch.linspace(
            0, oversampled_interval_length, steps=oversampling_factor
        )
        per_frequency_deltas -= oversampled_interval_length / 2
        per_frequency_deltas = einops.rearrange(per_frequency_deltas, "os -> os 1")
        fftfreq_grid = fftfreq_grid + per_frequency_deltas
    fftfreq_grid = fftfreq_grid / pixel_size
    fftfreq_grid_squared = fftfreq_grid**2
    n4 = fftfreq_grid_squared**2
    
    # calculate ctf
    ctf = -torch.sin(k1 * fftfreq_grid_squared * defocus + k2 * n4 - k3 - k5)
    if torch.any(k4 < 0):
        idx = (k4 < 0).squeeze(dim=(-2,-1)) 
        ctf[idx] *= torch.exp(k4[idx] * n4[idx])
    ctf = einops.reduce(ctf, "... os k -> ... k", reduction="mean") # oversampling reduction
    return ctf
