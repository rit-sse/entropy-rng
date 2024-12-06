#!/usr/bin/env python3
"""
IQ Data Bandpass Filter CLI Tool with Noise Floor Detection

This script applies a bandpass filter to IQ data stored in binary format. It supports:
- Automated detection of noise floor to dynamically set the bandpass filter parameters.
- Manual specification of filter parameters.
- Processing files containing unsigned 8-bit IQ samples.

Features:
- Bandpass filtering using the SciPy library.
- Validation of frequency parameters.
- Output of filtered IQ data to a specified file.

Example Usage:
--------------
1. Automatically detect noise floor and apply filter:
   ./filter_iq_data.py iq_samples.bin filtered_noise.bin --auto --fs 2048000 --order 5

2. Apply manual bandpass filter:
   ./filter_iq_data.py iq_samples.bin filtered_noise.bin --lowcut 200000 --highcut 800000 --fs 2048000 --order 5

3. Help Command:
   ./filter_iq_data.py --help

Arguments:
----------
file_path       Path to the input binary file containing IQ data.
output_file     Path to save the filtered IQ data.
--auto          Automatically detect noise floor and set filter parameters.
--lowcut        Low cutoff frequency (Hz) (manual setting, overrides auto-detection).
--highcut       High cutoff frequency (Hz) (manual setting, overrides auto-detection).
--fs            Sampling frequency (Hz). Default is 2048000 Hz.
--order         Order of the Butterworth filter. Default is 5.

Author: Ryan
Date: 2024-11-29
"""

import argparse
import numpy as np
from scipy.signal import butter, filtfilt


def compute_spectrum(data: np.ndarray, fs: float) -> tuple:
    """
    Compute the power spectrum of the IQ data.

    Args:
        data (np.ndarray): Input complex IQ data.
        fs (float): Sampling frequency (Hz).

    Returns:
        tuple: Frequencies and power spectrum.
    """
    spectrum = np.fft.fftshift(np.fft.fft(data))
    power_spectrum = 10 * np.log10(np.abs(spectrum) ** 2)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(data), d=1/fs))
    return freqs, power_spectrum


def detect_noise_floor(data: np.ndarray, fs: float, threshold_db: float = None) -> tuple:
    """
    Automatically detect noise floor and determine filter cutoff frequencies.

    Args:
        data (np.ndarray): Input complex IQ data.
        fs (float): Sampling frequency (Hz).
        threshold_db (float): Threshold in dB for identifying the noise floor.
                              If None, use a dynamic threshold based on the spectrum.

    Returns:
        tuple: Low and high cutoff frequencies (Hz).
    """
    freqs, power_spectrum = compute_spectrum(data, fs)

    # Compute dynamic threshold if not provided
    if threshold_db is None:
        threshold_db = np.median(power_spectrum) - 10  # 10 dB below the median power
        print(f"Dynamic threshold set to {threshold_db:.2f} dB.")

    # Identify noise floor frequencies
    noise_indices = np.where(power_spectrum < threshold_db)[0]

    # Fallback: Use the middle 80% of the spectrum if no clear noise floor
    if len(noise_indices) == 0:
        print("No clear noise floor detected. Falling back to default range.")
        lowcut = freqs[int(len(freqs) * 0.1)]  # 10% of the range
        highcut = freqs[int(len(freqs) * 0.9)]  # 90% of the range
    else:
        lowcut = np.min(freqs[noise_indices])
        highcut = np.max(freqs[noise_indices])

    print(f"Detected noise floor: Lowcut = {lowcut:.2f} Hz, Highcut = {highcut:.2f} Hz")
    return lowcut, highcut



def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Apply a bandpass filter to the provided data.

    Args:
        data (np.ndarray): Input complex IQ data.
        lowcut (float): Low cutoff frequency (Hz).
        highcut (float): High cutoff frequency (Hz).
        fs (float): Sampling frequency (Hz).
        order (int): Order of the Butterworth filter.

    Returns:
        np.ndarray: Bandpass-filtered data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Debugging: Print normalized frequencies
    print(f"Lowcut (normalized): {low}, Highcut (normalized): {high}")

    # Validate critical frequencies
    if not (0 < low < 1 and 0 < high < 1):
        raise ValueError("Critical frequencies must be between 0 and Nyquist frequency.")

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def filter_iq_data(file_path: str, output_file: str, fs: float, order: int, auto: bool, lowcut: float = None, highcut: float = None) -> None:
    """
    Read, filter, and save IQ data from a binary file.

    Args:
        file_path (str): Path to the input binary file containing IQ data.
        output_file (str): Path to save the filtered IQ data.
        fs (float): Sampling frequency (Hz).
        order (int): Order of the Butterworth filter.
        auto (bool): Automatically detect noise floor if True.
        lowcut (float): Low cutoff frequency (Hz) (optional).
        highcut (float): High cutoff frequency (Hz) (optional).
    """
    # Load IQ data as unsigned 8-bit integers and center them
    iq_data = np.fromfile(file_path, dtype=np.uint8)
    iq_data = iq_data - 127.5  # Center data around 0
    I = iq_data[0::2]  # Extract I (in-phase) samples
    Q = iq_data[1::2]  # Extract Q (quadrature) samples

    # Combine into complex IQ samples
    iq_complex = I + 1j * Q

    # Auto-detect or use manual filter settings
    if auto:
        lowcut, highcut = detect_noise_floor(iq_complex, fs)
    elif lowcut is None or highcut is None:
        raise ValueError("Manual lowcut and highcut must be provided if auto-detection is disabled.")

    # Apply bandpass filter
    filtered = bandpass_filter(iq_complex, lowcut, highcut, fs, order)

    # Save filtered IQ data
    filtered.tofile(output_file)
    print(f"Filtered IQ data saved to {output_file}.")


def main() -> None:
    """
    Parse command-line arguments and execute the filtering process.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Apply a bandpass filter to IQ data stored in binary format, "
            "with options for automated noise floor detection or manual settings.\n\n"
            "Example Usage:\n"
            "1. Automatically detect noise floor and apply filter:\n"
            "   ./filter_iq_data.py iq_samples.bin filtered_noise.bin --auto --fs 2048000 --order 5\n\n"
            "2. Apply manual bandpass filter:\n"
            "   ./filter_iq_data.py iq_samples.bin filtered_noise.bin --lowcut 200000 --highcut 800000 --fs 2048000 --order 5\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the input binary file containing IQ data."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the filtered IQ data."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically detect noise floor and set filter parameters."
    )
    parser.add_argument(
        "--lowcut",
        type=float,
        help="Low cutoff frequency (Hz). Overrides auto-detection."
    )
    parser.add_argument(
        "--highcut",
        type=float,
        help="High cutoff frequency (Hz). Overrides auto-detection."
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=2048000,
        help="Sampling frequency (Hz). Default is 2048000 Hz."
    )
    parser.add_argument(
        "--order",
        type=int,
        default=5,
        help="Order of the Butterworth filter. Default is 5."
    )

    args = parser.parse_args()

    # Execute filtering with provided arguments
    filter_iq_data(
        file_path=args.file_path,
        output_file=args.output_file,
        fs=args.fs,
        order=args.order,
        auto=args.auto,
        lowcut=args.lowcut,
        highcut=args.highcut
    )


if __name__ == "__main__":
    main()
