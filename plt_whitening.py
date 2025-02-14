"""
Plotting whitening filter results
"""
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import torch

from torch_fourier_filter.whitening import whitening_filter


def plot_whitening_spectrum_smoothing_comparison(mrc_path):
    """
    Plotting whitening filter results
    """
    # Load MRC file
    with mrcfile.open(mrc_path) as mrc:
        image = mrc.data

    # If 3D, take central slice
    if len(image.shape) == 3:
        image = image[image.shape[0] // 2]

    # Convert to torch tensor
    image_tensor = torch.from_numpy(image).float()

    # Compute FFT
    image_dft = torch.fft.rfft2(image_tensor)
    image_dft[0, 0] = 0 + 0j
    max_freq = 0.5

    # Define different smoothing parameters to test
    smoothing_params = [
        {"sigma": 2.0, "kernel": 21},
        {"sigma": 3.0, "kernel": 21},
        {"sigma": 10.0, "kernel": 101},
        {"sigma": 1.0, "kernel": 3},
        {"sigma": 1.0, "kernel": 7},
        {"sigma": 1.0, "kernel": 11},
    ]

    # Create two subplots: one for sigma comparison, one for kernel size comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # First plot unsmoothed filter on both plots
    filter_result, filter_1d, frequency_1d = whitening_filter(
        image_dft=image_dft,
        rfft=True,
        max_freq=max_freq,
        output_shape=image.shape,
        smooth_filter=False,  # No smoothing
        return_1d=True,
    )

    filter_np = filter_1d.cpu().numpy()
    freq_axis = frequency_1d.cpu().numpy()

    # Add unsmoothed to both plots
    ax1.plot(freq_axis, filter_np, "--", label="Unsmoothed", color="black")
    ax2.plot(freq_axis, filter_np, "--", label="Unsmoothed", color="black")

    # Plot different sigma values (fixed kernel size = 5)
    for params in smoothing_params[:3]:
        filter_result, filter_1d, frequency_1d = whitening_filter(
            image_dft=image_dft,
            rfft=True,
            max_freq=max_freq,  # Set a fixed max_freq
            output_shape=image.shape,
            smooth_filter=True,
            smooth_sigma=params["sigma"],
            smooth_kernel_size=params["kernel"],  # Fixed kernel size
            return_1d=True,
        )

        filter_np = filter_1d.cpu().numpy()
        freq_axis = frequency_1d.cpu().numpy()

        ax1.plot(freq_axis, filter_np, label=f'sigma={params["sigma"]}, kernel=5')

    ax1.set_xlabel("Frequency (relative to Nyquist)")
    ax1.set_ylabel("Filter magnitude")
    ax1.set_title("Effect of Different Sigma Values (Fixed Kernel Size = 5)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, max_freq - 0.01)
    ax1.set_ylim(0, 1.01)

    # First plot unsmoothed filter on both plots
    filter_result, filter_1d, frequency_1d = whitening_filter(
        image_dft=image_dft,
        rfft=True,
        max_freq=max_freq,
        output_shape=image.shape,
        smooth_filter=False,  # No smoothing
        return_1d=True,
    )

    filter_np = filter_1d.cpu().numpy()
    freq_axis = frequency_1d.cpu().numpy()

    # Add unsmoothed to both plots
    ax1.plot(freq_axis, filter_np, "--", label="Unsmoothed", color="black")
    ax2.plot(freq_axis, filter_np, "--", label="Unsmoothed", color="black")

    # Plot different kernel sizes (fixed sigma = 1.0)
    for params in smoothing_params[3:]:
        filter_result, filter_1d, frequency_1d = whitening_filter(
            image_dft=image_dft,
            rfft=True,
            max_freq=max_freq,  # Set a fixed max_freq
            output_shape=image.shape,
            smooth_filter=True,
            smooth_sigma=1.0,  # Fixed sigma
            smooth_kernel_size=params["kernel"],
            return_1d=True,
        )

        filter_np = filter_1d.cpu().numpy()
        freq_axis = frequency_1d.cpu().numpy()

        ax2.plot(freq_axis, filter_np, label=f'sigma=1.0, kernel={params["kernel"]}')

    ax2.set_xlabel("Frequency (relative to Nyquist)")
    ax2.set_ylabel("Filter magnitude")
    ax2.set_title("Effect of Different Kernel Sizes (Fixed Sigma = 1.0)")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, max_freq - 0.01)
    ax2.set_ylim(0, 1.01)
    plt.tight_layout()
    plt.show()

    # Also plot the input image and its FFT for reference
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot original image
    im1 = ax1.imshow(image, cmap="gray")
    ax1.set_title("Original Image")
    plt.colorbar(im1, ax=ax1)

    # Plot FFT magnitude (log scale)
    fft_mag = np.log(np.abs(image_dft.cpu().numpy() ** 2) + 1)
    fft_mag = (
        (fft_mag - fft_mag.min()) * 255 / (fft_mag.max() - fft_mag.min())
    ).astype(np.uint8)
    im2 = ax2.imshow(fft_mag, cmap="viridis")
    ax2.set_title("Log FFT Magnitude")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.show()
    # After the existing plots, add whitening verification
    # Apply 2D whitening filter to image
    whitened_dft = image_dft * filter_result
    whitened_image = torch.fft.irfft2(whitened_dft)

    # Compute FFT of whitened image
    whitened_dft_verify = torch.fft.rfft2(whitened_image)
    whitened_dft_verify[0, 0] = 0 + 0j

    # Calculate 1D whitening filter from whitened image
    filter_result_verify, filter_1d_verify, frequency_1d_verify = whitening_filter(
        image_dft=whitened_dft_verify,
        rfft=True,
        max_freq=max_freq,
        output_shape=image.shape,
        smooth_filter=False,
        return_1d=True,
    )

    # Create verification plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        frequency_1d_verify.cpu().numpy(),
        filter_1d_verify.cpu().numpy(),
        label="Whitening filter of whitened image",
    )
    plt.xlabel("Frequency (relative to Nyquist)")
    plt.ylabel("Filter magnitude")
    plt.title("Verification of Whitening (should be approximately flat)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, max_freq - 0.01)
    plt.ylim(0, 1.01)
    plt.tight_layout()
    plt.show()

    # Also plot the whitened image and its FFT for reference
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot whitened image
    im1 = ax1.imshow(whitened_image.cpu().numpy(), cmap="gray")
    ax1.set_title("Whitened Image")
    plt.colorbar(im1, ax=ax1)

    # Plot FFT magnitude of whitened image (log scale)
    whitened_fft_mag = np.log(np.abs(whitened_dft_verify.cpu().numpy() ** 2) + 1)
    whitened_fft_mag = (
        (whitened_fft_mag - whitened_fft_mag.min())
        * 255
        / (whitened_fft_mag.max() - whitened_fft_mag.min())
    ).astype(np.uint8)
    im2 = ax2.imshow(whitened_fft_mag, cmap="viridis")
    ax2.set_title("Log FFT Magnitude of Whitened Image")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.show()


def radial_profile(data, center, angles):
    y, x = np.indices(data.shape)  # first determine radii of all pixels
    # yx = complex(y, x) #make complex
    all_angles = (
        np.angle(x - center[0] + 1j * (y - center[1]), deg=True) + 180
    )  # get angles
    # excluded_indices = np.argwhere(all_angles < angles[0]) #find indices outside this angular range
    # to test

    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    ind = np.argsort(r.flat)  # get sorted indices
    sr = r.flat[ind]  # sorted radii

    sangles = all_angles.flat[ind]  # sort angles in same way
    # included_indices = np.argwhere(sangles > angles[0])
    # print(angles[0])
    # print(angles[1])
    excluded_indices = np.argwhere(sangles < int(angles[0]))
    excluded_indices2 = np.argwhere(sangles > int(angles[1]))

    sim = data.flat[ind]  # image values sorted by radii
    sim[excluded_indices] = 0  # image values sorted by radii
    sim[excluded_indices2] = 0  # image values sorted by radii
    ri = sr.astype(np.int32)  # integer part of radii (bin size = 1)
    # determining distance between changes
    deltar = ri[1:] - ri[:-1]  # assume all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number in radius bin
    csim = np.cumsum(
        sim, dtype=np.float64
    )  # cumulative sum to figure out sums for each radii bin
    tbin = csim[rind[1:]] - csim[rind[:-1]]  # sum for image values in radius bins
    radialprofile = tbin / nr  # the answer
    # smoothed = savgol_filter(radialprofile, 99, 2)
    the_r = sr[rind[:-1]]

    # return [abs(smoothed), the_r]
    return [abs(radialprofile), the_r]


def plot_power_spectrum(mrc_path):
    # Load MRC file
    with mrcfile.open(mrc_path) as mrc:
        image = mrc.data

    # If 3D, take central slice
    if len(image.shape) == 3:
        image = image[image.shape[0] // 2]

    # Convert to torch tensor and compute full 2D FFT
    image_tensor = torch.from_numpy(image).float()
    image_fft = torch.fft.fft2(image_tensor)
    image_fft[0, 0] = 0 + 0j

    # Compute power spectrum
    power_spectrum = torch.abs(image_fft) ** 2
    power_spectrum = power_spectrum.cpu().numpy()

    # FFT shift to center the zero frequency
    power_spectrum = np.fft.fftshift(power_spectrum)

    # Calculate center coordinates
    center = (power_spectrum.shape[0] // 2, power_spectrum.shape[1] // 2)

    # Compute radial profile (using full 360 degrees)
    radial_avg, radii = radial_profile(power_spectrum, center, [0, 360])

    # Convert radii to Nyquist fractions
    nyquist_radii = radii / (power_spectrum.shape[0] // 2)

    # radial average sqrt
    radial_avg = np.sqrt(radial_avg)

    # recipricol
    radial_avg = 1 / radial_avg

    # divide by max value
    radial_avg = radial_avg / radial_avg.max()

    # recipricol
    # radial_avg = 1/radial_avg

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot power spectrum (log scale)
    power_spectrum_log = np.log10(power_spectrum + 1)
    im1 = ax1.imshow(power_spectrum_log, cmap="viridis")
    ax1.set_title("Log Power Spectrum")
    plt.colorbar(im1, ax=ax1)

    # Plot radial average
    ax2.plot(nyquist_radii, radial_avg)
    ax2.set_xlabel("Spatial Frequency (fraction of Nyquist)")
    ax2.set_ylabel("Power")
    ax2.set_title("Radial Average of Power Spectrum")
    ax2.grid(True)
    ax2.set_xlim(0, np.sqrt(2) / 2)  # Maximum frequency at corners is sqrt(2) * Nyquist

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Replace with your MRC file path
    mrc_path = "/Users/josh/git/benchmark_template_matching_methods/data/xenon_225_000_0.0_DW.mrc"
    plot_whitening_spectrum_smoothing_comparison(mrc_path)
    plot_power_spectrum(mrc_path)
