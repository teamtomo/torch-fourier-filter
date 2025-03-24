import torch

from torch_fourier_filter.dose_weight import (
    critical_exposure,
    critical_exposure_bfactor,
    cumulative_dose_filter_3d,
)


def test_critical_exposure():
    fft_freq = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    expected_output = torch.tensor([14.1383, 6.3823, 4.6287, 3.9365, 3.5869])
    output = critical_exposure(fft_freq)
    assert torch.allclose(
        output, expected_output, atol=1e-3
    ), "critical_exposure output mismatch"


def test_critical_exposure_bfactor():
    fft_freq = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    bfac = 1.0
    expected_output = torch.tensor([400.0, 100.0, 44.444, 25.0, 16.0])
    output = critical_exposure_bfactor(fft_freq, bfac)
    assert torch.allclose(
        output, expected_output, atol=1e-3
    ), "critical_exposure_bfactor output mismatch"


def test_cumulative_dose_filter_3d():
    # Test parameters
    volume_shape = (32, 32, 32)
    start_exposure = 0.0
    end_exposure = 10.0
    pixel_size = 1.0
    crit_exposure_bfactor = -1
    rfft = True
    fftshift = False

    # Call the function
    dose_filter = cumulative_dose_filter_3d(
        volume_shape=volume_shape,
        pixel_size=pixel_size,
        start_exposure=start_exposure,
        end_exposure=end_exposure,
        crit_exposure_bfactor=crit_exposure_bfactor,
        rfft=rfft,
        fftshift=fftshift,
    )

    # Check if the values are within a reasonable range
    # TODO: Test these against known, static values rather than just range
    assert torch.all(dose_filter >= 0) and torch.all(
        dose_filter <= 1
    ), "Dose filter values out of range"

    # Test with different bfac values
    # TODO: Test these against known, static values rather than just range
    crit_bfac_values = [0.5, 1.0, 2.0]
    for bfac in crit_bfac_values:
        dose_filter = cumulative_dose_filter_3d(
            volume_shape=volume_shape,
            pixel_size=pixel_size,
            start_exposure=start_exposure,
            end_exposure=end_exposure,
            crit_exposure_bfactor=bfac,
            rfft=rfft,
            fftshift=fftshift,
        )
        assert torch.all(dose_filter >= 0) and torch.all(
            dose_filter <= 1
        ), f"Dose filter values out of range for bfac={bfac}"
