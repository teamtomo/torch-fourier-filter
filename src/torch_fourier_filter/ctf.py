"""CTF module for Fourier filtering."""

import einops
import numpy as np
import torch
from scipy import constants as C
from torch_grid_utils.fftfreq_grid import fftfreq_grid


def calculate_relativistic_electron_wavelength(
    energy: float | torch.Tensor,
) -> torch.Tensor:
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


def calculate_relativistic_gamma(
    energy: float | torch.Tensor,
) -> torch.Tensor:
    """Calculate the relativistic Lorentz factor (gamma).

    The Lorentz factor is defined as:
    γ = 1 + eV/(m₀c²)

    where:
    - e is the elementary charge
    - V is the acceleration potential
    - m₀ is the electron rest mass
    - c is the speed of light

    Parameters
    ----------
    energy : float | torch.Tensor
        Acceleration potential in volts.

    Returns
    -------
    gamma : torch.Tensor
        Relativistic Lorentz factor (dimensionless).
    """
    c = C.speed_of_light
    m0 = C.electron_mass
    e = C.elementary_charge
    V = torch.as_tensor(energy, dtype=torch.float)

    # γ = 1 + eV/(m₀c²)
    gamma = 1 + (e * V) / (m0 * c**2)
    return gamma


def calculate_relativistic_beta(
    energy: float | torch.Tensor,
) -> torch.Tensor:
    """Calculate the relativistic beta factor (v/c).

    The beta factor is defined as:
    β = v/c = √(1 - 1/γ²)

    where γ is the Lorentz factor and v is the electron velocity.

    Parameters
    ----------
    energy : float | torch.Tensor
        Acceleration potential in volts.

    Returns
    -------
    beta : torch.Tensor
        Relativistic beta factor (dimensionless, v/c).
    """
    gamma = calculate_relativistic_gamma(energy)

    # β = √(1 - 1/γ²) = √((γ² - 1)/γ²)
    beta = torch.sqrt((gamma**2 - 1) / gamma**2)
    return beta


def calculate_defocus_phase_aberration(
    defocus_um: torch.Tensor,
    voltage_kv: torch.Tensor,
    spherical_aberration_mm: torch.Tensor,
    fftfreq_grid_angstrom_squared: torch.Tensor,
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
    return (
        k1 * fftfreq_grid_angstrom_squared * defocus
        + k2 * fftfreq_grid_angstrom_squared**2
    )


def calculate_additional_phase_shift(
    phase_shift_degrees: torch.Tensor,
) -> torch.Tensor:
    return torch.deg2rad(phase_shift_degrees)


def calculate_amplitude_contrast_equivalent_phase_shift(
    amplitude_contrast_fraction: torch.Tensor,
) -> torch.Tensor:
    return torch.arctan(
        amplitude_contrast_fraction / torch.sqrt(1 - amplitude_contrast_fraction**2)
    )


def calculate_total_phase_shift(
    defocus_um: torch.Tensor,
    voltage_kv: torch.Tensor,
    spherical_aberration_mm: torch.Tensor,
    phase_shift_degrees: torch.Tensor,
    amplitude_contrast_fraction: torch.Tensor,
    fftfreq_grid_angstrom_squared: torch.Tensor,
) -> torch.Tensor:
    """Calculate the total phase shift for the CTF.

    Parameters
    ----------
    defocus_um : torch.Tensor
        Defocus in micrometers, positive is underfocused.
    voltage_kv : torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration_mm : torch.Tensor
        Spherical aberration in millimeters (mm).
    phase_shift_degrees : torch.Tensor
        Phase shift in degrees.
    amplitude_contrast_fraction : torch.Tensor
        Amplitude contrast as a fraction (0 to 1).
    fftfreq_grid_angstrom_squared : torch.Tensor
        Precomputed squared frequency grid in Angstroms^-2.

    Returns
    -------
    total_phase_shift : torch.Tensor
        The total phase shift for the given parameters.
    """
    phase_aberration = calculate_defocus_phase_aberration(
        defocus_um, voltage_kv, spherical_aberration_mm, fftfreq_grid_angstrom_squared
    )

    additional_phase_shift = calculate_additional_phase_shift(phase_shift_degrees)

    amplitude_contrast_phase_shift = (
        calculate_amplitude_contrast_equivalent_phase_shift(amplitude_contrast_fraction)
    )

    return phase_aberration - additional_phase_shift - amplitude_contrast_phase_shift


# Below will be LPP related funcations.
# These should ultimately be moved to a separate file
# Not doing now as this will all be ripped out for torch-ctf


def initialize_laser_params(
    NA: float,
    laser_wavelength_angstrom: float,
) -> tuple[float, float]:
    """
    Initialize the laser parameters.

    Parameters
    ----------
    NA: float
        Numerical aperture of the objective lens.
    laser_wavelength_angstrom: float
        Wavelength of the laser.

    Returns
    -------
    beam_waist: float
        Beam waist of the laser.
    rayleigh_range: float
        Rayleigh range of the laser.
    """
    beam_waist = laser_wavelength_angstrom / (np.pi * NA)
    rayleigh_range = beam_waist / NA
    return beam_waist, rayleigh_range


def make_laser_coords(
    fft_freq_grid_angstrom: torch.Tensor,
    electron_wavelength_angstrom: float,
    focal_length_angstrom: float,
    laser_xy_angle_deg: float,
    laser_long_offset_angstrom: float,
    laser_trans_offset_angstrom: float,
    beam_waist_angstroms: float,
    rayleigh_range_angstroms: float,
) -> torch.Tensor:
    """
    Make the laser coordinates for the CTF.

    Parameters
    ----------
    fft_freq_grid_angstrom : torch.Tensor
        FFT frequency grid in Angstroms^-1, shape (..., H, W, 2) where last dim is [x, y].
    electron_wavelength_angstrom : float
        Electron wavelength in Angstroms.
    focal_length_angstrom : float
        Focal length in Angstroms.
    laser_xy_angle_deg : float
        Laser rotation angle in degrees.
    laser_long_offset_angstrom : float
        Longitudinal offset in Angstroms.
    laser_trans_offset_angstrom : float
        Transverse offset in Angstroms.
    beam_waist_angstroms : float
        Beam waist (w0) in Angstroms.
    rayleigh_range_angstroms : float
        Rayleigh range (zR) in Angstroms.

    Returns
    -------
    laser_coords : torch.Tensor
        Dimensionless laser coordinates, shape (..., H, W, 2) where last dim is [Lx, Ly].
    """
    # Convert frequency coordinates to physical coordinates [A]
    physical_freq_coords = (
        fft_freq_grid_angstrom * electron_wavelength_angstrom * focal_length_angstrom
    )

    # Create rotation matrix for xy_angle
    angle_rad = torch.deg2rad(
        torch.tensor(laser_xy_angle_deg, device=physical_freq_coords.device)
    )
    cos_angle = torch.cos(angle_rad)
    sin_angle = torch.sin(angle_rad)

    # Rotation matrix: [[cos, -sin], [sin, cos]]
    rotation_matrix = torch.tensor(
        [[cos_angle, -sin_angle], [sin_angle, cos_angle]],
        device=physical_freq_coords.device,
        dtype=physical_freq_coords.dtype,
    )

    # Apply rotation: R @ coords (broadcasting over spatial dimensions)
    # physical_freq_coords is (..., H, W, 2), rotation_matrix is (2, 2)
    rotated_coords = torch.einsum(
        "...ij,jk->...ik", physical_freq_coords, rotation_matrix
    )

    # Apply translation offsets
    offset_tensor = torch.tensor(
        [laser_long_offset_angstrom, laser_trans_offset_angstrom],
        device=physical_freq_coords.device,
        dtype=physical_freq_coords.dtype,
    )
    translated_coords = rotated_coords - offset_tensor

    # Make dimensionless coordinates
    scale_tensor = torch.tensor(
        [rayleigh_range_angstroms, beam_waist_angstroms],
        device=physical_freq_coords.device,
        dtype=physical_freq_coords.dtype,
    )
    laser_coords = translated_coords / scale_tensor

    return laser_coords


def get_eta(
    eta0: float | torch.Tensor,
    laser_coords: torch.Tensor,
    beta: float | torch.Tensor,
    NA: float | torch.Tensor,
    pol_angle_deg: float | torch.Tensor,
    xz_angle_deg: float | torch.Tensor,
    laser_phi_deg: float | torch.Tensor,
) -> torch.Tensor:
    """
    Calculate eta (phase modulation) due to laser standing wave.

    Parameters
    ----------
    eta0 : float | torch.Tensor
        Base eta value.
    laser_coords : torch.Tensor
        Dimensionless laser coordinates from make_laser_coords,
        shape (..., H, W, 2) where last dim is [Lx, Ly].
    beta : float | torch.Tensor
        Beta parameter from scope.
    NA : float | torch.Tensor
        Numerical aperture of the laser.
    pol_angle_deg : float | torch.Tensor
        Polarization angle in degrees.
    xz_angle_deg : float | torch.Tensor
        XZ angle in degrees.
    laser_phi_deg : float | torch.Tensor
        Laser phi in degrees.

    Returns
    -------
    eta : torch.Tensor
        Phase modulation due to laser standing wave.
    """
    # Extract Lx and Ly from laser coordinates tensor
    Lx = laser_coords[..., 0]  # (..., H, W)
    Ly = laser_coords[..., 1]  # (..., H, W)

    # Convert parameters to tensors with proper device/dtype
    device = laser_coords.device
    dtype = laser_coords.dtype

    eta0 = torch.as_tensor(eta0, device=device, dtype=dtype)
    beta = torch.as_tensor(beta, device=device, dtype=dtype)
    NA = torch.as_tensor(NA, device=device, dtype=dtype)
    pol_angle_rad = torch.deg2rad(
        torch.as_tensor(pol_angle_deg, device=device, dtype=dtype)
    )
    xz_angle_rad = torch.deg2rad(
        torch.as_tensor(xz_angle_deg, device=device, dtype=dtype)
    )
    laser_phi_rad = torch.deg2rad(
        torch.as_tensor(laser_phi_deg, device=device, dtype=dtype)
    )

    # Calculate intermediate terms
    Lx_squared_plus_1 = 1 + Lx**2
    Ly_squared = Ly**2

    # Main calculation following the original formula
    # eta0/2 * exp(-2*Ly^2/(1+Lx^2)) / sqrt(1+Lx^2)
    base_term = (
        (eta0 / 2)
        * torch.exp(-2 * Ly_squared / Lx_squared_plus_1)
        / torch.sqrt(Lx_squared_plus_1)
    )

    # Calculate the complex modulation term
    # (1-2*beta^2*cos^2(pol_angle))
    pol_modulation = 1 - 2 * beta**2 * torch.cos(pol_angle_rad) ** 2

    # exp(-xz_angle^2 * (2/NA^2) * (1+Lx^2))
    xz_exp_term = torch.exp(-(xz_angle_rad**2) * (2 / NA**2) * Lx_squared_plus_1)

    # (1+Lx^2)^(-1/4)
    power_term = Lx_squared_plus_1 ** (-0.25)

    # cos(2*Lx*Ly^2/(1+Lx^2) + 4*Lx/NA^2 - 1.5*arctan(Lx) - laser_phi)
    phase_arg = (
        2 * Lx * Ly_squared / Lx_squared_plus_1
        + 4 * Lx / NA**2
        - 1.5 * torch.arctan(Lx)
        - laser_phi_rad
    )
    cos_term = torch.cos(phase_arg)

    # Combine all terms
    modulation_term = 1 + pol_modulation * xz_exp_term * power_term * cos_term

    return base_term * modulation_term


def get_eta0_from_peak_phase_deg(
    peak_phase_deg: float | torch.Tensor,
    laser_coords: torch.Tensor,
    beta: float | torch.Tensor,
    NA: float | torch.Tensor,
    pol_angle_deg: float | torch.Tensor,
    xz_angle_deg: float | torch.Tensor,
    laser_phi_deg: float | torch.Tensor,
) -> torch.Tensor:
    """
    Calculate eta0 from desired peak phase in degrees.

    This function iteratively determines the eta0 value needed to achieve
    a desired peak phase, accounting for the fact that the maximum phase
    may not occur at the expected location due to tilts and other effects.

    Parameters
    ----------
    peak_phase_deg : float | torch.Tensor
        Desired peak phase in degrees.
    laser_coords : torch.Tensor
        Dimensionless laser coordinates from make_laser_coords, shape (..., H, W, 2).
    beta : float | torch.Tensor
        Beta parameter from scope.
    NA : float | torch.Tensor
        Numerical aperture of the laser.
    pol_angle_deg : float | torch.Tensor
        Polarization angle in degrees.
    xz_angle_deg : float | torch.Tensor
        XZ angle in degrees.
    laser_phi_deg : float | torch.Tensor
        Laser phi in degrees.

    Returns
    -------
    eta0 : torch.Tensor
        Calibrated eta0 value in radians.
    """
    # Convert peak phase to radians for initial guess
    device = laser_coords.device
    dtype = laser_coords.dtype

    peak_phase_deg = torch.as_tensor(peak_phase_deg, device=device, dtype=dtype)
    eta0_test = torch.deg2rad(peak_phase_deg)  # [rad]

    # Calculate eta with the test eta0 value
    eta_test = get_eta(
        eta0=eta0_test,
        laser_coords=laser_coords,
        beta=beta,
        NA=NA,
        pol_angle_deg=pol_angle_deg,
        xz_angle_deg=xz_angle_deg,
        laser_phi_deg=laser_phi_deg,
    )

    # Find the actual peak phase achieved with this eta0
    peak_phase_deg_test = torch.rad2deg(eta_test.max())  # [deg]

    # Scale eta0 to achieve the desired peak phase
    # eta0_corrected = eta0_test * (desired_peak / actual_peak)
    eta0 = eta0_test * peak_phase_deg / peak_phase_deg_test  # [rad]

    return eta0


def calc_LPP_phase(
    fft_freq_grid: torch.Tensor,
    NA: float,
    laser_wavelength_angstrom: float,
    focal_length_angstrom: float,
    laser_xy_angle_deg: float,
    laser_xz_angle_deg: float,
    laser_long_offset_angstrom: float,
    laser_trans_offset_angstrom: float,
    laser_polarization_angle_deg: float,
    peak_phase_deg: float,
    voltage: float | torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the laser phase plate phase modulation.

    Parameters
    ----------
    fft_freq_grid : torch.Tensor
        FFT frequency grid in cycles/Å, shape (..., H, W, 2).
    NA : float
        Numerical aperture of the laser.
    laser_wavelength_angstrom : float
        Wavelength of the laser in Angstroms.
    focal_length_angstrom : float
        Focal length in Angstroms.
    laser_xy_angle_deg : float
        Laser rotation angle in the xy plane in degrees.
    laser_xz_angle_deg : float
        Laser angle in the xz plane in degrees.
    laser_long_offset_angstrom : float
        Longitudinal offset of the laser in Angstroms.
    laser_trans_offset_angstrom : float
        Transverse offset of the laser in Angstroms.
    laser_polarization_angle_deg : float
        Polarization angle of the laser in degrees.
    peak_phase_deg : float
        Desired peak phase in degrees.
    voltage : float | torch.Tensor
        Acceleration voltage in kilovolts (kV).

    Returns
    -------
    eta : torch.Tensor
        Laser phase modulation in radians.
    """
    # Calculate laser parameters
    beam_waist_angstroms, rayleigh_range_angstroms = initialize_laser_params(
        NA, laser_wavelength_angstrom
    )
    beta = calculate_relativistic_beta(voltage * 1e3)  # Convert kV to V
    electron_wavelength_angstrom = (
        calculate_relativistic_electron_wavelength(voltage * 1e3) * 1e10
    )  # Convert m to Å

    # Make laser coordinates
    laser_coords = make_laser_coords(
        fft_freq_grid,
        electron_wavelength_angstrom,
        focal_length_angstrom,
        laser_xy_angle_deg,
        laser_long_offset_angstrom,
        laser_trans_offset_angstrom,
        beam_waist_angstroms,
        rayleigh_range_angstroms,
    )

    # Calculate laser phase (antinode configuration)
    laser_phi = 0  # antinode, 90 is node
    eta0 = get_eta0_from_peak_phase_deg(
        peak_phase_deg,
        laser_coords,
        beta,
        NA,
        laser_polarization_angle_deg,
        laser_xz_angle_deg,
        laser_phi,
    )
    eta = get_eta(
        eta0,
        laser_coords,
        beta,
        NA,
        laser_polarization_angle_deg,
        laser_xz_angle_deg,
        laser_phi,
    )

    return eta


def calc_LPP_ctf_2D(
    defocus: float | torch.Tensor,
    astigmatism: float | torch.Tensor,
    astigmatism_angle: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    image_shape: tuple[int, int],
    rfft: bool,
    fftshift: bool,
    # Laser parameters
    NA: float,
    laser_wavelength_angstrom: float,
    focal_length_angstrom: float,
    laser_xy_angle_deg: float,
    laser_xz_angle_deg: float,
    laser_long_offset_angstrom: float,
    laser_trans_offset_angstrom: float,
    laser_polarization_angle_deg: float,
    peak_phase_deg: float,
) -> torch.Tensor:
    """Calculate the Laser Phase Plate (LPP) modified Contrast Transfer Function (CTF) for a 2D image.

    This function is similar to calculate_ctf_2d but uses laser parameters to generate
    a spatially varying phase shift instead of a uniform phase shift.

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
    pixel_size : float | torch.Tensor
        Pixel size in Angströms per pixel (Å px⁻¹).
    image_shape : tuple[int, int]
        Shape of 2D images onto which CTF will be applied.
    rfft : bool
        Generate the CTF containing only the non-redundant half transform from a rfft.
    fftshift : bool
        Whether to apply fftshift on the resulting CTF images.
    NA : float
        Numerical aperture of the laser.
    laser_wavelength_angstrom : float
        Wavelength of the laser in Angstroms.
    focal_length_angstrom : float
        Focal length in Angstroms.
    laser_xy_angle_deg : float
        Laser rotation angle in the xy plane in degrees.
    laser_xz_angle_deg : float
        Laser angle in the xz plane in degrees.
    laser_long_offset_angstrom : float
        Longitudinal offset of the laser in Angstroms.
    laser_trans_offset_angstrom : float
        Transverse offset of the laser in Angstroms.
    laser_polarization_angle_deg : float
        Polarization angle of the laser in degrees.
    peak_phase_deg : float
        Desired peak phase in degrees.

    Returns
    -------
    ctf : torch.Tensor
        The Laser Phase Plate modified Contrast Transfer Function for the given parameters.
    """
    if isinstance(defocus, torch.Tensor):
        device = defocus.device
    else:
        device = torch.device("cpu")

    # Convert parameters to torch.Tensor with proper device
    defocus = torch.as_tensor(defocus, dtype=torch.float, device=device)
    astigmatism = torch.as_tensor(astigmatism, dtype=torch.float, device=device)
    astigmatism_angle = torch.as_tensor(
        astigmatism_angle, dtype=torch.float, device=device
    )
    pixel_size = torch.as_tensor(pixel_size, dtype=torch.float, device=device)
    voltage = torch.as_tensor(voltage, dtype=torch.float, device=device)
    spherical_aberration = torch.as_tensor(
        spherical_aberration, dtype=torch.float, device=device
    )
    amplitude_contrast = torch.as_tensor(
        amplitude_contrast, dtype=torch.float, device=device
    )
    image_shape = torch.as_tensor(image_shape, dtype=torch.int, device=device)

    # Reshape for broadcasting
    defocus = einops.rearrange(defocus, "... -> ... 1 1")
    voltage = einops.rearrange(voltage, "... -> ... 1 1")
    spherical_aberration = einops.rearrange(spherical_aberration, "... -> ... 1 1")
    amplitude_contrast = einops.rearrange(amplitude_contrast, "... -> ... 1 1")

    # Construct 2D frequency grids and rescale cycles / px -> cycles / Å
    fft_freq_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=False,
        device=device,
    )
    fft_freq_grid = fft_freq_grid / einops.rearrange(pixel_size, "... -> ... 1 1 1")
    fft_freq_grid_squared = einops.reduce(
        fft_freq_grid**2, "... f->...", reduction="sum"
    )

    # Calculate laser phase using the dedicated function
    laser_phase_radians = calc_LPP_phase(
        fft_freq_grid=fft_freq_grid,
        NA=NA,
        laser_wavelength_angstrom=laser_wavelength_angstrom,
        focal_length_angstrom=focal_length_angstrom,
        laser_xy_angle_deg=laser_xy_angle_deg,
        laser_xz_angle_deg=laser_xz_angle_deg,
        laser_long_offset_angstrom=laser_long_offset_angstrom,
        laser_trans_offset_angstrom=laser_trans_offset_angstrom,
        laser_polarization_angle_deg=laser_polarization_angle_deg,
        peak_phase_deg=peak_phase_deg,
        voltage=voltage,
    )

    # Convert laser phase from radians to degrees for compatibility with calculate_total_phase_shift
    laser_phase_degrees = torch.rad2deg(laser_phase_radians)

    # Calculate the astigmatism vector (same as in calculate_ctf_2d)
    sin_theta = torch.sin(torch.deg2rad(astigmatism_angle))
    cos_theta = torch.cos(torch.deg2rad(astigmatism_angle))
    unit_astigmatism_vector_yx = einops.rearrange(
        [sin_theta, cos_theta], "yx ... -> ... yx"
    )
    astigmatism = einops.rearrange(astigmatism, "... -> ... 1")
    # Multiply with the square root of astigmatism so to get the right amplitude after squaring later
    astigmatism_vector = torch.sqrt(astigmatism) * unit_astigmatism_vector_yx
    # Calculate unitvectors from the frequency grids
    # Reuse already computed fft_freq_grid_squared to avoid redundant pow operations
    fft_freq_grid_norm = torch.sqrt(
        einops.rearrange(fft_freq_grid_squared, "... -> ... 1")
        + torch.finfo(torch.float32).eps
    )
    direction_unitvector = fft_freq_grid / fft_freq_grid_norm
    # Subtract the astigmatism from the defocus
    defocus -= einops.rearrange(astigmatism, "... -> ... 1")
    # Add the squared dotproduct between the direction unitvector and the astigmatism vector
    defocus = (
        defocus
        + einops.einsum(
            direction_unitvector, astigmatism_vector, "... h w f, ... f -> ... h w"
        )
        ** 2
        * 2
    )

    # Calculate CTF using laser phase instead of uniform phase shift
    ctf = -torch.sin(
        calculate_total_phase_shift(
            defocus_um=defocus,
            voltage_kv=voltage,
            spherical_aberration_mm=spherical_aberration,
            phase_shift_degrees=laser_phase_degrees,  # Use spatially varying laser phase
            amplitude_contrast_fraction=amplitude_contrast,
            fftfreq_grid_angstrom_squared=fft_freq_grid_squared,
        )
    )

    return ctf


def calculate_ctf_2d(
    defocus: float | torch.Tensor,
    astigmatism: float | torch.Tensor,
    astigmatism_angle: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
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
    astigmatism_angle = torch.as_tensor(
        astigmatism_angle, dtype=torch.float, device=device
    )
    pixel_size = torch.as_tensor(pixel_size, dtype=torch.float, device=device)
    voltage = torch.as_tensor(voltage, dtype=torch.float, device=device)
    spherical_aberration = torch.as_tensor(
        spherical_aberration, dtype=torch.float, device=device
    )
    amplitude_contrast = torch.as_tensor(
        amplitude_contrast, dtype=torch.float, device=device
    )
    phase_shift = torch.as_tensor(phase_shift, dtype=torch.float, device=device)
    image_shape = torch.as_tensor(image_shape, dtype=torch.int, device=device)

    defocus = einops.rearrange(defocus, "... -> ... 1 1")
    voltage = einops.rearrange(voltage, "... -> ... 1 1")
    spherical_aberration = einops.rearrange(spherical_aberration, "... -> ... 1 1")
    amplitude_contrast = einops.rearrange(amplitude_contrast, "... -> ... 1 1")
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
    fft_freq_grid_squared = einops.reduce(
        fft_freq_grid**2, "... f->...", reduction="sum"
    )

    # Calculate the astigmatism vector
    sin_theta = torch.sin(torch.deg2rad(astigmatism_angle))
    cos_theta = torch.cos(torch.deg2rad(astigmatism_angle))
    unit_astigmatism_vector_yx = einops.rearrange(
        [sin_theta, cos_theta], "yx ... -> ... yx"
    )
    astigmatism = einops.rearrange(astigmatism, "... -> ... 1")
    # Multiply with the square root of astigmatism so to get the right amplitude after squaring later
    astigmatism_vector = torch.sqrt(astigmatism) * unit_astigmatism_vector_yx
    # Calculate unitvectors from the frequency grids
    # Reuse already computed fft_freq_grid_squared to avoid redundant pow operations
    fft_freq_grid_norm = torch.sqrt(
        einops.rearrange(fft_freq_grid_squared, "... -> ... 1")
        + torch.finfo(torch.float32).eps
    )
    direction_unitvector = fft_freq_grid / fft_freq_grid_norm
    # Subtract the astigmatism from the defocus
    defocus -= einops.rearrange(astigmatism, "... -> ... 1")
    # Add the squared dotproduct between the direction unitvector and the astigmatism vector
    defocus = (
        defocus
        + einops.einsum(
            direction_unitvector, astigmatism_vector, "... h w f, ... f -> ... h w"
        )
        ** 2
        * 2
    )
    # calculate ctf
    ctf = -torch.sin(
        calculate_total_phase_shift(
            defocus_um=defocus,
            voltage_kv=voltage,
            spherical_aberration_mm=spherical_aberration,
            phase_shift_degrees=phase_shift,
            amplitude_contrast_fraction=amplitude_contrast,
            fftfreq_grid_angstrom_squared=fft_freq_grid_squared,
        )
    )

    return ctf


def calculate_ctf_1d(
    defocus: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
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
    spherical_aberration = torch.as_tensor(
        spherical_aberration, dtype=torch.float, device=device
    )
    amplitude_contrast = torch.as_tensor(
        amplitude_contrast, dtype=torch.float, device=device
    )
    phase_shift = torch.as_tensor(phase_shift, dtype=torch.float, device=device)
    pixel_size = torch.as_tensor(pixel_size, dtype=torch.float, device=device)

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
    expansion_string = "... -> ... " + " ".join(["1"] * fftfreq_grid.ndim)

    pixel_size = einops.rearrange(pixel_size, expansion_string)
    defocus = einops.rearrange(defocus, expansion_string)
    voltage = einops.rearrange(voltage, expansion_string)
    spherical_aberration = einops.rearrange(spherical_aberration, expansion_string)
    phase_shift = einops.rearrange(phase_shift, expansion_string)
    amplitude_contrast = einops.rearrange(amplitude_contrast, expansion_string)

    fftfreq_grid = fftfreq_grid / pixel_size

    # calculate ctf
    ctf = -torch.sin(
        calculate_total_phase_shift(
            defocus_um=defocus,
            voltage_kv=voltage,
            spherical_aberration_mm=spherical_aberration,
            phase_shift_degrees=phase_shift,
            amplitude_contrast_fraction=amplitude_contrast,
            fftfreq_grid_angstrom_squared=fftfreq_grid**2,
        )
    )

    if oversampling_factor > 1:
        # reduce oversampling
        ctf = einops.reduce(
            ctf, "... os k -> ... k", reduction="mean"
        )  # oversampling reduction
    return ctf
