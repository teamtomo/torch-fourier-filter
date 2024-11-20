import torch

from torch_fourier_filter.dose_weight import (
    critical_exposure,
    critical_exposure_Bfac,
    cumulative_dose_filter_3d,
)


def test_critical_exposure():
    fft_freq = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    expected_output = torch.tensor([14.1383, 6.3823, 4.6287, 3.9365, 3.5869])
    output = critical_exposure(fft_freq)
    assert torch.allclose(
        output, expected_output, atol=1e-3
    ), "critical_exposure output mismatch"


def test_critical_exposure_Bfac():
    fft_freq = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    Bfac = 1.0
    expected_output = torch.tensor([200.0, 50.0, 22.222, 12.5, 8.0])
    output = critical_exposure_Bfac(fft_freq, Bfac)
    assert torch.allclose(
        output, expected_output, atol=1e-3
    ), "critical_exposure_Bfac output mismatch"


def test_cumulative_dose_filter_3d():
    # Test parameters
    volume_shape = (32, 32, 32)
    num_frames = 10
    start_exposure = 0.0
    pixel_size = 1.0
    flux = 1.0
    Bfac = -1.0
    rfft = True
    fftshift = False

    # Call the function
    dose_filter = cumulative_dose_filter_3d(
        volume_shape=volume_shape,
        num_frames=num_frames,
        start_exposure=start_exposure,
        pixel_size=pixel_size,
        flux=flux,
        Bfac=Bfac,
        rfft=rfft,
        fftshift=fftshift,
    )

    # Check if the values are within a reasonable range
    assert torch.all(dose_filter >= 0) and torch.all(
        dose_filter <= 1
    ), "Dose filter values out of range"

    # Test with different Bfac values
    Bfac_values = [0.5, 1.0, 2.0]
    for Bfac in Bfac_values:
        dose_filter = cumulative_dose_filter_3d(
            volume_shape=volume_shape,
            num_frames=num_frames,
            start_exposure=start_exposure,
            pixel_size=pixel_size,
            flux=flux,
            Bfac=Bfac,
            rfft=rfft,
            fftshift=fftshift,
        )
        assert torch.all(dose_filter >= 0) and torch.all(
            dose_filter <= 1
        ), f"Dose filter values out of range for Bfac={Bfac}"
