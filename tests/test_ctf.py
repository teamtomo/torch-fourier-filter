import torch

from torch_fourier_filter.ctf import (
    calc_LPP_ctf_2D,
    calc_LPP_phase,
    calculate_ctf_1d,
    calculate_ctf_2d,
    calculate_relativistic_beta,
    calculate_relativistic_electron_wavelength,
    calculate_relativistic_gamma,
    get_eta,
    initialize_laser_params,
    make_laser_coords,
)

EXPECTED_2D = torch.tensor(
    [
        [
            [
                0.1000,
                0.2427,
                0.6287,
                0.9862,
                0.6624,
                -0.5461,
                0.6624,
                0.9862,
                0.6287,
                0.2427,
            ],
            [
                0.2427,
                0.3802,
                0.7344,
                0.9998,
                0.5475,
                -0.6611,
                0.5475,
                0.9998,
                0.7344,
                0.3802,
            ],
            [
                0.6287,
                0.7344,
                0.9519,
                0.9161,
                0.1449,
                -0.9151,
                0.1449,
                0.9161,
                0.9519,
                0.7344,
            ],
            [
                0.9862,
                0.9998,
                0.9161,
                0.4211,
                -0.5461,
                -0.9531,
                -0.5461,
                0.4211,
                0.9161,
                0.9998,
            ],
            [
                0.6624,
                0.5475,
                0.1449,
                -0.5461,
                -0.9998,
                -0.2502,
                -0.9998,
                -0.5461,
                0.1449,
                0.5475,
            ],
            [
                -0.5461,
                -0.6611,
                -0.9151,
                -0.9531,
                -0.2502,
                0.8651,
                -0.2502,
                -0.9531,
                -0.9151,
                -0.6611,
            ],
            [
                0.6624,
                0.5475,
                0.1449,
                -0.5461,
                -0.9998,
                -0.2502,
                -0.9998,
                -0.5461,
                0.1449,
                0.5475,
            ],
            [
                0.9862,
                0.9998,
                0.9161,
                0.4211,
                -0.5461,
                -0.9531,
                -0.5461,
                0.4211,
                0.9161,
                0.9998,
            ],
            [
                0.6287,
                0.7344,
                0.9519,
                0.9161,
                0.1449,
                -0.9151,
                0.1449,
                0.9161,
                0.9519,
                0.7344,
            ],
            [
                0.2427,
                0.3802,
                0.7344,
                0.9998,
                0.5475,
                -0.6611,
                0.5475,
                0.9998,
                0.7344,
                0.3802,
            ],
        ],
        [
            [
                0.1000,
                0.3351,
                0.8755,
                0.7628,
                -0.7326,
                -0.1474,
                -0.7326,
                0.7628,
                0.8755,
                0.3351,
            ],
            [
                0.3351,
                0.5508,
                0.9657,
                0.5861,
                -0.8741,
                0.0932,
                -0.8741,
                0.5861,
                0.9657,
                0.5508,
            ],
            [
                0.8755,
                0.9657,
                0.8953,
                -0.0979,
                -0.9766,
                0.7290,
                -0.9766,
                -0.0979,
                0.8953,
                0.9657,
            ],
            [
                0.7628,
                0.5861,
                -0.0979,
                -0.9648,
                -0.1474,
                0.8998,
                -0.1474,
                -0.9648,
                -0.0979,
                0.5861,
            ],
            [
                -0.7326,
                -0.8741,
                -0.9766,
                -0.1474,
                0.9995,
                -0.5378,
                0.9995,
                -0.1474,
                -0.9766,
                -0.8741,
            ],
            [
                -0.1474,
                0.0932,
                0.7290,
                0.8998,
                -0.5378,
                -0.3948,
                -0.5378,
                0.8998,
                0.7290,
                0.0932,
            ],
            [
                -0.7326,
                -0.8741,
                -0.9766,
                -0.1474,
                0.9995,
                -0.5378,
                0.9995,
                -0.1474,
                -0.9766,
                -0.8741,
            ],
            [
                0.7628,
                0.5861,
                -0.0979,
                -0.9648,
                -0.1474,
                0.8998,
                -0.1474,
                -0.9648,
                -0.0979,
                0.5861,
            ],
            [
                0.8755,
                0.9657,
                0.8953,
                -0.0979,
                -0.9766,
                0.7290,
                -0.9766,
                -0.0979,
                0.8953,
                0.9657,
            ],
            [
                0.3351,
                0.5508,
                0.9657,
                0.5861,
                -0.8741,
                0.0932,
                -0.8741,
                0.5861,
                0.9657,
                0.5508,
            ],
        ],
    ]
)


def test_1d_ctf_single():
    result = calculate_ctf_1d(
        defocus=1.5,
        pixel_size=8,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        n_samples=10,
        oversampling_factor=3,
    )
    expected = torch.tensor(
        [
            0.1033,
            0.1476,
            0.2784,
            0.4835,
            0.7271,
            0.9327,
            0.9794,
            0.7389,
            0.1736,
            -0.5358,
        ]
    )
    assert torch.allclose(result, expected, atol=1e-4)


def test_1d_ctf_batch():
    result = calculate_ctf_1d(
        defocus=[[[1.5, 2.5]]],
        pixel_size=[[[8, 8]]],
        voltage=[[[300, 300]]],
        spherical_aberration=[[[2.7, 2.7]]],
        amplitude_contrast=[[[0.1, 0.1]]],
        phase_shift=[[[0, 0]]],
        n_samples=10,
        oversampling_factor=1,
    )
    expected = torch.tensor(
        [
            [
                0.1000,
                0.1444,
                0.2755,
                0.4819,
                0.7283,
                0.9385,
                0.9903,
                0.7519,
                0.1801,
                -0.5461,
            ],
            [
                0.1000,
                0.1738,
                0.3880,
                0.6970,
                0.9617,
                0.9237,
                0.3503,
                -0.5734,
                -0.9877,
                -0.1474,
            ],
        ]
    )
    assert result.shape == (1, 1, 2, 10)
    assert torch.allclose(result, expected, atol=1e-4)


def test_calculate_relativistic_electron_wavelength():
    """Check function matches expected value from literature.

    De Graef, Marc (2003-03-27).
    Introduction to Conventional Transmission Electron Microscopy.
    Cambridge University Press. doi:10.1017/cbo9780511615092
    """
    result = calculate_relativistic_electron_wavelength(300e3)
    expected = 1.969e-12
    assert abs(result - expected) < 1e-15


def test_2d_ctf_batch():
    result = calculate_ctf_2d(
        defocus=[[[1.5, 2.5]]],
        astigmatism=[[[0, 0]]],
        astigmatism_angle=[[[0, 0]]],
        pixel_size=[[[8, 8]]],
        voltage=[[[300, 300]]],
        spherical_aberration=[[[2.7, 2.7]]],
        amplitude_contrast=[[[0.1, 0.1]]],
        phase_shift=[[[0, 0]]],
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
    )
    expected = EXPECTED_2D
    assert result.shape == (1, 1, 2, 10, 10)
    assert torch.allclose(result[0, 0], expected, atol=1e-4)


def test_2d_ctf_astigmatism():
    result = calculate_ctf_2d(
        defocus=[2.0, 2.0, 2.5, 2.0],
        astigmatism=[0.5, 1.0, 0.5, 0.5],
        astigmatism_angle=[0, 30, 45, 90],
        pixel_size=8,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
    )
    assert result.shape == (4, 10, 10)

    # First case:
    # Along the X axis the powerspectrum should be like the 2.5 um defocus one
    assert torch.allclose(result[0, 0, :], EXPECTED_2D[1][0, :], atol=1e-4)
    # Along the Y axis the powerspectrum should be like the 1.5 um defocus one
    assert torch.allclose(result[0, :, 0], EXPECTED_2D[0][:, 0], atol=1e-4)

    # Second case:
    # At 30 degrees, X and Y should get half of the astigmatism (cos(60)=0.5), so we still
    # get the same powerspectrum along the axes as in the first case, since the astigmatism is double.
    assert torch.allclose(result[1, 0, :], EXPECTED_2D[1][0, :], atol=1e-4)
    assert torch.allclose(result[1, :, 0], EXPECTED_2D[0][:, 0], atol=1e-4)

    # Third case:
    # At 45 degrees, the powerspectrum should be the same in X and Y and exactly
    # the average defocus (2.5)
    assert torch.allclose(result[2, 0, :], EXPECTED_2D[1][0, :], atol=1e-4)
    assert torch.allclose(result[2, :, 0], EXPECTED_2D[1][:, 0], atol=1e-4)

    # Fourth case:
    # At 90 degrees, we should get 2.5 um defocus in the Y axis
    # and 1.5 um defocus in the X axis.
    assert torch.allclose(result[3, 0, :], EXPECTED_2D[0][0, :], atol=1e-4)
    assert torch.allclose(result[3, :, 0], EXPECTED_2D[1][:, 0], atol=1e-4)


def test_calculate_relativistic_gamma():
    """Test relativistic gamma calculation."""
    # For 300 kV electrons
    result = calculate_relativistic_gamma(300e3)
    expected = 1.587  # Known value for 300 kV electrons
    assert abs(result - expected) < 0.001


def test_calculate_relativistic_beta():
    """Test relativistic beta calculation."""
    # For 300 kV electrons
    result = calculate_relativistic_beta(300e3)
    expected = 0.776  # Known value for 300 kV electrons (v/c)
    assert abs(result - expected) < 0.001


def test_initialize_laser_params():
    """Test laser parameter initialization."""
    NA = 0.05
    laser_wavelength_angstrom = 1064e4  # 1064 nm

    beam_waist, rayleigh_range = initialize_laser_params(NA, laser_wavelength_angstrom)

    # Check that beam waist and rayleigh range are positive and reasonable
    assert beam_waist > 0
    assert rayleigh_range > 0
    assert (
        rayleigh_range > beam_waist
    )  # Rayleigh range should be larger than beam waist


def test_make_laser_coords():
    """Test laser coordinate transformation."""
    # Create a simple frequency grid
    fft_freq_grid = torch.zeros(1, 10, 10, 2)
    fft_freq_grid[0, :, :, 0] = torch.linspace(-0.5, 0.5, 10).unsqueeze(
        0
    )  # x frequencies
    fft_freq_grid[0, :, :, 1] = torch.linspace(-0.5, 0.5, 10).unsqueeze(
        1
    )  # y frequencies

    # Test parameters
    electron_wavelength_angstrom = 1.97e-2  # ~300 kV
    focal_length_angstrom = 20e7  # 20 mm
    laser_xy_angle_deg = 0.0
    laser_long_offset_angstrom = 0.0
    laser_trans_offset_angstrom = 0.0
    beam_waist_angstroms = 1e6
    rayleigh_range_angstroms = 1e7

    result = make_laser_coords(
        fft_freq_grid,
        electron_wavelength_angstrom,
        focal_length_angstrom,
        laser_xy_angle_deg,
        laser_long_offset_angstrom,
        laser_trans_offset_angstrom,
        beam_waist_angstroms,
        rayleigh_range_angstroms,
    )

    # Check output shape and properties
    assert result.shape == (1, 10, 10, 2)
    assert torch.isfinite(result).all()


def test_get_eta():
    """Test eta calculation with known parameters."""
    # Create simple laser coordinates
    laser_coords = torch.zeros(1, 5, 5, 2)
    laser_coords[0, :, :, 0] = torch.linspace(-1, 1, 5).unsqueeze(0)  # Lx
    laser_coords[0, :, :, 1] = torch.linspace(-1, 1, 5).unsqueeze(1)  # Ly

    # Test parameters
    eta0 = 1.0
    beta = 0.776  # 300 kV electrons
    NA = 0.05
    pol_angle_deg = 0.0
    xz_angle_deg = 0.0
    laser_phi_deg = 0.0

    result = get_eta(
        eta0, laser_coords, beta, NA, pol_angle_deg, xz_angle_deg, laser_phi_deg
    )

    # Check output properties
    assert result.shape == (1, 5, 5)
    assert torch.isfinite(result).all()
    assert result.max() <= eta0  # Should not exceed eta0


def test_calc_LPP_phase():
    """Test LPP phase calculation with proven parameters."""
    # Create frequency grid
    fft_freq_grid = torch.zeros(1, 8, 8, 2)
    fft_freq_grid[0, :, :, 0] = torch.linspace(-0.1, 0.1, 8).unsqueeze(0)
    fft_freq_grid[0, :, :, 1] = torch.linspace(-0.1, 0.1, 8).unsqueeze(1)

    # Proven parameters from your test script
    result = calc_LPP_phase(
        fft_freq_grid=fft_freq_grid,
        NA=0.05,
        laser_wavelength_angstrom=1064e4,
        focal_length_angstrom=20e7,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
        voltage=300.0,
    )

    # Check output properties
    assert result.shape == (1, 8, 8)
    assert torch.isfinite(result).all()
    assert result.max() > 0  # Should have positive phase values


def test_calc_LPP_ctf_2D():
    """Test full LPP CTF calculation with proven parameters."""
    result = calc_LPP_ctf_2D(
        defocus=1.0,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        pixel_size=1.0,
        image_shape=(16, 16),
        rfft=False,
        fftshift=True,
        # Laser parameters (proven from test script)
        NA=0.05,
        laser_wavelength_angstrom=1064e4,
        focal_length_angstrom=20e7,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
    )

    # Check output properties
    assert result.shape == (16, 16)
    assert torch.isfinite(result).all()
    assert (
        result.min() >= -1.0 and result.max() <= 1.0
    )  # CTF values should be in [-1, 1]


def test_calc_LPP_ctf_2D_batch():
    """Test LPP CTF with batched inputs."""
    defocus_values = [0.5, 1.0, 1.5]

    result = calc_LPP_ctf_2D(
        defocus=defocus_values,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        pixel_size=1.0,
        image_shape=(8, 8),
        rfft=False,
        fftshift=True,
        # Laser parameters
        NA=0.05,
        laser_wavelength_angstrom=1064e4,
        focal_length_angstrom=20e7,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
    )

    # Check batched output
    assert result.shape == (3, 8, 8)
    assert torch.isfinite(result).all()

    # Different defocus values should produce different CTFs
    assert not torch.allclose(result[0], result[1], atol=1e-3)
    assert not torch.allclose(result[1], result[2], atol=1e-3)


def test_lpp_vs_standard_ctf_difference():
    """Test that LPP CTF differs from standard CTF due to laser modulation."""
    # Common parameters
    common_params = {
        "defocus": 1.0,
        "astigmatism": 0.0,
        "astigmatism_angle": 0.0,
        "voltage": 300.0,
        "spherical_aberration": 2.7,
        "amplitude_contrast": 0.07,
        "pixel_size": 1.0,
        "image_shape": (16, 16),
        "rfft": False,
        "fftshift": True,
    }

    # Standard CTF
    standard_ctf = calculate_ctf_2d(phase_shift=0.0, **common_params)

    # LPP CTF
    lpp_ctf = calc_LPP_ctf_2D(
        NA=0.05,
        laser_wavelength_angstrom=1064e4,
        focal_length_angstrom=20e7,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
        **common_params,
    )

    # They should be different due to laser modulation
    assert not torch.allclose(standard_ctf, lpp_ctf, atol=1e-3)

    # But both should be valid CTFs
    assert torch.isfinite(standard_ctf).all()
    assert torch.isfinite(lpp_ctf).all()
