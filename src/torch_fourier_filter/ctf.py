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
    energy : float | torch.Tensor
        Acceleration potential in volts.

    Returns
    -------
    wavelength : float | torch.Tensor
        Relativistic wavelength of the electron in meters.
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

def calculate_phase_aberration(
    defocus_um: torch.Tensor,
    voltage_kv: torch.Tensor,
    spherical_aberration_mm: torch.Tensor,
    phase_shift: torch.Tensor,
    amplitude_contrast: torch.Tensor,
    fftfreq_grid_angstrom_squared: torch.Tensor
) -> torch.Tensor:
    """Calculate the phase aberration.

    Parameters
    ----------
    defocus_um : torch.Tensor
        Defocus in micrometers, positive is underfocused.
    voltage_kv : torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration_mm : torch.Tensor
        Spherical aberration in millimeters (mm).
    phase_shift : torch.Tensor
        Angle of phase shift applied to CTF in degrees.
    amplitude_contrast : torch.Tensor
        Fraction of amplitude contrast (value in range [0, 1]).
    fftfreq_grid_angstrom_squared : torch.Tensor
        Precomputed squared frequency grid in Angstroms^-2.

    Returns
    -------
    phase_aberration : torch.Tensor
        The phase aberration for the given parameters.
    """

    # Unit conversions
    defocus = defocus_um * 1e4  # micrometers -> angstroms
    voltage = voltage_kv * 1e3  # kV -> V
    spherical_aberration = spherical_aberration_mm * 1e7  # mm -> angstroms

    # derived quantities used in CTF calculation
    _lambda = (
        calculate_relativistic_electron_wavelength(voltage) * 1e10
    )  # meters -> angstroms

    k1 = -C.pi * _lambda
    k2 = C.pi / 2 * spherical_aberration * _lambda**3
    k3 = torch.deg2rad(torch.as_tensor(phase_shift, dtype=torch.float))
    k5 = torch.arctan(amplitude_contrast / torch.sqrt(1 - amplitude_contrast**2))

    return k1 * fftfreq_grid_angstrom_squared * defocus + k2 * fftfreq_grid_angstrom_squared ** 2 - k3 - k5

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
    defocus : float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
        `(defocus_u + defocus_v) / 2`
    astigmatism : float | torch.Tensor
        Amount of astigmatism in micrometers.
        `(defocus_u - defocus_v) / 2`
    astigmatism_angle : float | torch.Tensor
        Angle of astigmatism in degrees. 0 places `defocus_u` along the y-axis.
    voltage : float | torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration : float | torch.Tensor
        Spherical aberration in millimeters (mm).
    amplitude_contrast : float | torch.Tensor
        Fraction of amplitude contrast (value in range [0, 1]).
    b_factor : float | torch.Tensor
        B-factor in square angstroms. Should be positive.
    phase_shift : float | torch.Tensor
        Angle of phase shift applied to CTF in degrees.
    pixel_size : float | torch.Tensor
        Pixel size in Angströms per pixel (Å px⁻¹).
    image_shape : tuple[int, int]
        Shape of 2D images onto which CTF will be applied.
    rfft : bool
        Generate the CTF containing only the non-redundant half transform from a rfft.
    fftshift : bool
        Whether to apply fftshift on the resulting CTF images.

    Returns
    -------
    ctf : torch.Tensor
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

    defocus = einops.rearrange(defocus, "... -> ... 1 1")
    voltage = einops.rearrange(voltage, "... -> ... 1 1")
    spherical_aberration = einops.rearrange(spherical_aberration, "... -> ... 1 1")
    amplitude_contrast = einops.rearrange(amplitude_contrast, "... -> ... 1 1")
    b_factor = einops.rearrange(b_factor, "... -> ... 1 1")
    phase_shift = einops.rearrange(phase_shift, "... -> ... 1 1")
    
    # construct 2D frequency grids and rescale cycles / px -> cycles / Å
    fft_freq_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=False,
        device=device,
    )
    fft_freq_grid = fft_freq_grid / einops.rearrange(pixel_size, "... -> ... 1 1 1")
    fft_freq_grid_squared = einops.reduce(fft_freq_grid**2,"... f->...", reduction="sum")
    # Calculate the astigmatism vector
    astigmatism_vector = torch.stack(
        [
            torch.sin(torch.deg2rad(astigmatism_angle)),
            torch.cos(torch.deg2rad(astigmatism_angle)),
        ],
        dim=-1,
    ) * torch.sqrt(einops.rearrange(astigmatism, "... -> ... 1"))
    # Calculate unitvectors from the frequency grids
    direction_unitvector = fft_freq_grid / (torch.norm(fft_freq_grid, dim=(-1), keepdim=True) + torch.finfo(torch.float32).eps)
    # Subtract the astigmatism from the defocus
    defocus -= einops.rearrange(astigmatism, "... -> ... 1 1")
    # Add the squared dotproduct between the direction unitvector and the astigmatism vector
    defocus = defocus + einops.einsum(direction_unitvector, astigmatism_vector, "... h w f, ... f -> ... h w")**2 * 2 
    # calculate ctf
    ctf = -torch.sin(
        calculate_phase_aberration(
            defocus_um=defocus,
            voltage_kv=voltage,
            spherical_aberration_mm=spherical_aberration,
            phase_shift=phase_shift,
            amplitude_contrast=amplitude_contrast,
            fftfreq_grid_angstrom_squared=fft_freq_grid_squared
        )
    )

    if torch.any(b_factor > 0):
        k4 = -b_factor / 4
        idx = (k4 < 0).squeeze(dim=(-2,-1)) 
        ctf[idx] *= torch.exp(k4[idx] * fft_freq_grid_squared[idx])
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
    """Calculate the Contrast Transfer Function (CTF) for a 1D signal.

    Parameters
    ----------
    defocus : float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
    voltage : float | torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration : float | torch.Tensor
        Spherical aberration in millimeters (mm).
    amplitude_contrast : float | torch.Tensor
        Fraction of amplitude contrast (value in range [0, 1]).
    b_factor : float | torch.Tensor
        B-factor in square angstroms. Should be positive.
    phase_shift : float | torch.Tensor
        Angle of phase shift applied to CTF in degrees.
    pixel_size : float | torch.Tensor
        Pixel size in Angströms per pixel (Å px⁻¹).
    n_samples : int
        Number of samples in CTF.
    oversampling_factor : int
        Factor by which to oversample the CTF.

    Returns
    -------
    ctf : torch.Tensor
        The Contrast Transfer Function for the given parameters.
    """
    if isinstance(defocus, torch.Tensor):
        device = defocus.device
    else:
        device = torch.device("cpu")
        
    # to torch.Tensor
    defocus = torch.as_tensor(defocus, dtype=torch.float, device=device)
    voltage = torch.as_tensor(voltage, dtype=torch.float, device=device)
    spherical_aberration = torch.as_tensor(spherical_aberration, dtype=torch.float, device=device)
    amplitude_contrast = torch.as_tensor(amplitude_contrast, dtype=torch.float, device=device)
    phase_shift = torch.as_tensor(phase_shift, dtype=torch.float, device=device)
    b_factor = torch.as_tensor(b_factor, dtype=torch.float, device = device)
    pixel_size = torch.as_tensor(pixel_size, dtype=torch.float, device = device)
    
    
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

    # Add singletary frequencies according to the dimensions of fftfreq_grid
    expansion_string = "... -> ... "+ " ".join(["1"] * fftfreq_grid.ndim)

    pixel_size = einops.rearrange(pixel_size, expansion_string)
    defocus = einops.rearrange(defocus, expansion_string)
    voltage = einops.rearrange(voltage, expansion_string)
    spherical_aberration = einops.rearrange(spherical_aberration, expansion_string)
    phase_shift = einops.rearrange(phase_shift, expansion_string)
    amplitude_contrast = einops.rearrange(amplitude_contrast, expansion_string)
    b_factor = einops.rearrange(b_factor, expansion_string)

    fftfreq_grid = fftfreq_grid / pixel_size
    
    # calculate ctf
    ctf = -torch.sin(
        calculate_phase_aberration(
            defocus_um=defocus,
            voltage_kv=voltage,
            spherical_aberration_mm=spherical_aberration,
            phase_shift=phase_shift,
            amplitude_contrast=amplitude_contrast,
            fftfreq_grid_angstrom_squared=fftfreq_grid**2
        )
    )
    if torch.any(b_factor > 0):
        k4 = -b_factor / 4
        if oversampling_factor > 1:
            idx = (k4 < 0).squeeze(dim=(-2,-1))
        else:
            idx = (k4 < 0).squeeze(dim=(-1))
        ctf[idx] *= torch.exp(k4[idx] * fftfreq_grid[idx]**2)
    if oversampling_factor > 1:
        # reduce oversampling
        ctf = einops.reduce(ctf, "... os k -> ... k", reduction="mean") # oversampling reduction
    return ctf
