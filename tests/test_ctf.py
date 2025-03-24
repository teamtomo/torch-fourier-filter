import torch

from torch_fourier_filter.ctf import (
    calculate_ctf_1d,
    calculate_ctf_2d,
    calculate_relativistic_electron_wavelength,
)


def test_1d_ctf_single():
    result = calculate_ctf_1d(
        defocus=1.5,
        pixel_size=8,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        b_factor=0,
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
        b_factor=[[[0, 0]]],
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
    assert (result.shape == (1,1,2,10))
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
        b_factor=[[[0, 0]]],
        phase_shift=[[[0, 0]]],
        image_shape=(10,10),
        rfft=False,
        fftshift=False,
    )
    expected = torch.tensor(
        [[[ 0.1000,  0.2427,  0.6287,  0.9862,  0.6624, -0.5461,  0.6624,  0.9862,  0.6287,  0.2427],
          [ 0.2427,  0.3802,  0.7344,  0.9998,  0.5475, -0.6611,  0.5475,  0.9998,  0.7344,  0.3802],
          [ 0.6287,  0.7344,  0.9519,  0.9161,  0.1449, -0.9151,  0.1449,  0.9161,  0.9519,  0.7344],
          [ 0.9862,  0.9998,  0.9161,  0.4211, -0.5461, -0.9531, -0.5461,  0.4211,  0.9161,  0.9998],
          [ 0.6624,  0.5475,  0.1449, -0.5461, -0.9998, -0.2502, -0.9998, -0.5461,  0.1449,  0.5475],
          [-0.5461, -0.6611, -0.9151, -0.9531, -0.2502,  0.8651, -0.2502, -0.9531, -0.9151, -0.6611],
          [ 0.6624,  0.5475,  0.1449, -0.5461, -0.9998, -0.2502, -0.9998, -0.5461,  0.1449,  0.5475],
          [ 0.9862,  0.9998,  0.9161,  0.4211, -0.5461, -0.9531, -0.5461,  0.4211,  0.9161,  0.9998],
          [ 0.6287,  0.7344,  0.9519,  0.9161,  0.1449, -0.9151,  0.1449,  0.9161,  0.9519,  0.7344],
          [ 0.2427,  0.3802,  0.7344,  0.9998,  0.5475, -0.6611,  0.5475,  0.9998,  0.7344,  0.3802]],
         [[ 0.1000,  0.3351,  0.8755,  0.7628, -0.7326, -0.1474, -0.7326,  0.7628,  0.8755,  0.3351],
          [ 0.3351,  0.5508,  0.9657,  0.5861, -0.8741,  0.0932, -0.8741,  0.5861,  0.9657,  0.5508],
          [ 0.8755,  0.9657,  0.8953, -0.0979, -0.9766,  0.7290, -0.9766, -0.0979,  0.8953,  0.9657],
          [ 0.7628,  0.5861, -0.0979, -0.9648, -0.1474,  0.8998, -0.1474, -0.9648, -0.0979,  0.5861],
          [-0.7326, -0.8741, -0.9766, -0.1474,  0.9995, -0.5378,  0.9995, -0.1474, -0.9766, -0.8741],
          [-0.1474,  0.0932,  0.7290,  0.8998, -0.5378, -0.3948, -0.5378,  0.8998,  0.7290,  0.0932],
          [-0.7326, -0.8741, -0.9766, -0.1474,  0.9995, -0.5378,  0.9995, -0.1474, -0.9766, -0.8741],
          [ 0.7628,  0.5861, -0.0979, -0.9648, -0.1474,  0.8998, -0.1474, -0.9648, -0.0979,  0.5861],
          [ 0.8755,  0.9657,  0.8953, -0.0979, -0.9766,  0.7290, -0.9766, -0.0979,  0.8953,  0.9657],
          [ 0.3351,  0.5508,  0.9657,  0.5861, -0.8741,  0.0932, -0.8741,  0.5861,  0.9657,  0.5508]]]
    )
    assert (result.shape == (1,1,2,10,10))
    assert torch.allclose(result[0,0], expected, atol=1e-4)
