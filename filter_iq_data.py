from scipy.signal import butter, filtfilt
import numpy as np


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist

    # Debugging: Print normalized frequencies
    print(f"Lowcut (normalized): {low}, Highcut (normalized): {high}")

    # Validate critical frequencies
    if not (0 < low < 1 and 0 < high < 1):
        raise ValueError(
            "Critical frequencies must be between 0 and Nyquist frequency."
        )

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def filter_iq_data(file_path, output_file, lowcut, highcut, fs):
    iq_data = np.fromfile(file_path, dtype=np.uint8)
    iq_data = iq_data - 127.5  # Center data
    I = iq_data[0::2]
    Q = iq_data[1::2]

    # Apply bandpass filter
    filtered = bandpass_filter(I + 1j * Q, lowcut, highcut, fs)
    filtered.tofile(output_file)
    print(f"Filtered IQ data saved to {output_file}.")


filter_iq_data(
    file_path="iq_samples.bin",
    output_file="filtered_noise.bin",
    lowcut=200000,
    highcut=800000,
    fs=2048000,
)
