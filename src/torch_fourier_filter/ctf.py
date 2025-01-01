"""CTF module for Fourier filtering."""

import einops
import torch
from scipy import constants as C
from torch_grid_utils.fftfreq_grid import fftfreq_grid, fftshift_2d


def calculate_relativistic_electron_wavelength(energy: float) -> float:
    """Calculate the relativistic electron wavelength in SI units.

    For derivation see:
    1.  Kirkland, E. J. Advanced Computing in Electron Microscopy.
        (Springer International Publishing, 2020). doi:10.1007/978-3-030-33260-0.

    2.  https://en.wikipedia.org/wiki/Electron_diffraction#Relativistic_theory

    Parameters
    ----------
    energy: float
        acceleration potential in volts.

    Returns
    -------
    wavelength: float
        relativistic wavelength of the electron in meters.
    """
    h = C.Planck
    c = C.speed_of_light
    m0 = C.electron_mass
    e = C.elementary_charge
    V = energy
    eV = e * V

    numerator = h * c
    denominator = (eV * (2 * m0 * c**2 + eV)) ** 0.5
    return float(numerator / denominator)


def calculate_ctf_2d(
    defocus: float | torch.Tensor,
    astigmatism: float | torch.Tensor,
    astigmatism_angle: float | torch.Tensor,
    voltage: float,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float,
    b_factor: float | torch.Tensor,
    phase_shift: float | torch.Tensor,
    pixel_size: float,
    image_shape: tuple[int, int],
    rfft: bool,
    fftshift: bool,
) -> torch.Tensor:
    """
    Calculate the Contrast Transfer Function (CTF) for a 2D image.

    Parameters
    ----------
    defocus: float
        Defocus in micrometers, positive is underfocused.
        `(defocus_u + defocus_v) / 2`
    astigmatism: float
        Amount of astigmatism in micrometers.
        `(defocus_u - defocus_v) / 2`
    astigmatism_angle: float
        Angle of astigmatism in degrees. 0 places `defocus_u` along the y-axis.
    pixel_size: float
        Pixel size in Angströms per pixel (Å px⁻¹).
    voltage: float
        Acceleration voltage in kilovolts (kV).
    spherical_aberration: float
        Spherical aberration in millimeters (mm).
    amplitude_contrast: float
        Fraction of amplitude contrast (value in range [0, 1]).
    b_factor: float
        B-factor in square angstroms.
    phase_shift: float
        Angle of phase shift applied to CTF in degrees.
    image_shape: Tuple[int, int]
        Shape of 2D images onto which CTF will be applied.
    rfft: bool
        Generate the CTF containing only the non-redundant half transform from a rfft.
        Only one of `rfft` and `fftshift` may be `True`.
    fftshift: bool
        Whether to apply fftshift on the resulting CTF images.

    Returns
    -------
    ctf: torch.Tensor
        The Contrast Transfer Function for the given parameters.
    """
    # to torch.Tensor and unit conversions
    cs_tensor = False
    defocus = torch.atleast_1d(torch.as_tensor(defocus, dtype=torch.float))
    defocus *= 1e4  # micrometers -> angstroms
    astigmatism = torch.atleast_1d(torch.as_tensor(astigmatism, dtype=torch.float))
    astigmatism *= 1e4  # micrometers -> angstroms
    astigmatism_angle = torch.atleast_1d(
        torch.as_tensor(astigmatism_angle, dtype=torch.float)
    )
    astigmatism_angle *= C.pi / 180  # degrees -> radians
    pixel_size = torch.atleast_1d(torch.as_tensor(pixel_size))
    voltage = torch.atleast_1d(torch.as_tensor(voltage, dtype=torch.float))
    voltage *= 1e3  # kV -> V
    spherical_aberration = torch.atleast_1d(
        torch.as_tensor(spherical_aberration, dtype=torch.float)
    )
    spherical_aberration *= 1e7  # mm -> angstroms
    if spherical_aberration.shape[0] > 1:
        cs_tensor = True
    image_shape = torch.as_tensor(image_shape)

    # derived quantities used in CTF calculation
    defocus_u = defocus + astigmatism
    defocus_v = defocus - astigmatism
    _lambda = (
        calculate_relativistic_electron_wavelength(voltage) * 1e10
    )  # meters -> angstroms
    k1 = -C.pi * _lambda
    k2 = C.pi / 2 * spherical_aberration * _lambda**3
    k3 = torch.deg2rad(torch.as_tensor(phase_shift, dtype=torch.float))
    k4 = -b_factor / 4
    amplitude_contrast = torch.as_tensor(amplitude_contrast, dtype=torch.float)
    k5 = torch.arctan(amplitude_contrast / torch.sqrt(1 - amplitude_contrast**2))

    # construct 2D frequency grids and rescale cycles / px -> cycles / Å
    fft_freq_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=False,
    )

    fft_freq_grid = fft_freq_grid / einops.rearrange(pixel_size, "b -> b 1 1 1")
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

    yy2, xx2 = einops.rearrange(fftfreq_grid_squared, "b h w freq -> freq b h w")
    xy = einops.reduce(fft_freq_grid, "b h w freq -> b h w", reduction="prod")
    n4 = (
        einops.reduce(fftfreq_grid_squared, "b h w freq -> b h w", reduction="sum") ** 2
    )

    Axx = c2 * defocus_u + s2 * defocus_v
    Axy = c * s * (defocus_u - defocus_v)
    Ayy = s2 * defocus_u + c2 * defocus_v
    if cs_tensor:
        Axx_x2 = einops.rearrange(Axx, "... -> ... 1 1 1") * xx2
        Axy_xy = einops.rearrange(Axy, "... -> ... 1 1 1") * xy
        Ayy_y2 = einops.rearrange(Ayy, "... -> ... 1 1 1") * yy2
        k2 = einops.rearrange(k2, "... -> 1 1 ... 1 1")
    else:
        Ayy_y2 = einops.rearrange(Ayy, "... -> ... 1 1") * yy2
        Axx_x2 = einops.rearrange(Axx, "... -> ... 1 1") * xx2
        Axy_xy = einops.rearrange(Axy, "... -> ... 1 1") * xy

    # calculate ctf
    ctf = -torch.sin(k1 * (Axx_x2 + (2 * Axy_xy) + Ayy_y2) + k2 * n4 - k3 - k5)
    if k4 > 0:
        ctf *= torch.exp(k4 * n4)
    if fftshift is True:
        ctf = fftshift_2d(ctf, rfft=rfft)
    return ctf


def calculate_ctf_1d(
    defocus: float,
    voltage: float,
    spherical_aberration: float,
    amplitude_contrast: float,
    b_factor: float,
    phase_shift: float,
    pixel_size: float,
    n_samples: int,
    oversampling_factor: int,
) -> torch.Tensor:
    """
    Calculate the Contrast Transfer Function (CTF) for a 1D signal.

    Parameters
    ----------
    defocus: float
        Defocus in micrometers, positive is underfocused.
    pixel_size: float
        Pixel size in Angströms per pixel (Å px⁻¹).
    voltage: float
        Acceleration voltage in kilovolts (kV).
    spherical_aberration: float
        Spherical aberration in millimeters (mm).
    amplitude_contrast: float
        Fraction of amplitude contrast (value in range [0, 1]).
    b_factor: float
        B-factor in square angstroms.
    phase_shift: float
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
    # to torch.Tensor and unit conversions
    defocus = torch.atleast_1d(torch.as_tensor(defocus, dtype=torch.float))
    defocus = defocus * 1e4  # micrometers -> angstroms
    defocus = einops.rearrange(defocus, "... -> ... 1")
    pixel_size = torch.atleast_1d(torch.as_tensor(pixel_size))
    voltage = torch.atleast_1d(torch.as_tensor(voltage, dtype=torch.float))
    voltage = voltage * 1e3  # kV -> V
    spherical_aberration = torch.atleast_1d(
        torch.as_tensor(spherical_aberration, dtype=torch.float)
    )
    spherical_aberration *= 1e7  # mm -> angstroms

    # derived quantities used in CTF calculation
    _lambda = (
        calculate_relativistic_electron_wavelength(voltage) * 1e10
    )  # meters -> angstroms
    k1 = -C.pi * _lambda
    k2 = C.pi / 2 * spherical_aberration * _lambda**3
    k3 = torch.deg2rad(torch.as_tensor(phase_shift, dtype=torch.float))
    k4 = -b_factor / 4
    amplitude_contrast = torch.as_tensor(amplitude_contrast, dtype=torch.float)
    k5 = torch.arctan(amplitude_contrast / torch.sqrt(1 - amplitude_contrast**2))

    # construct frequency vector and rescale cycles / px -> cycles / Å
    fftfreq_grid = torch.linspace(0, 0.5, steps=n_samples)  # (n_samples, )

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
        defocus = einops.rearrange(defocus, "... -> ... 1")  # oversampling dim

    fftfreq_grid = fftfreq_grid / pixel_size
    fftfreq_grid_squared = fftfreq_grid**2
    n4 = fftfreq_grid_squared**2

    # calculate ctf
    ctf = -torch.sin(k1 * fftfreq_grid_squared * defocus + k2 * n4 - k3 - k5)
    if k4 > 0:
        ctf *= torch.exp(k4 * n4)

    if oversampling_factor > 1:
        ctf = einops.reduce(ctf, "... os k -> ... k", reduction="mean")
    return ctf
