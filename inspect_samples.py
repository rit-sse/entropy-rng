import numpy as np
import matplotlib.pyplot as plt

# Step 1: Check File Existence
try:
    iq_data = np.fromfile("iq_samples.bin", dtype=np.uint8)
except FileNotFoundError:
    print("File not found. Ensure 'iq_samples.bin' is in the same directory.")
    exit()

# Step 2: Verify Data Loaded
print(f"Data loaded: {len(iq_data)} samples")
if len(iq_data) == 0:
    print("File appears empty. Verify the IQ capture process.")
    exit()

# Step 3: Center the Data
iq_data = iq_data - 127.5  # Center around 0
I = iq_data[0::2]
Q = iq_data[1::2]
print(f"Loaded {len(I)} I samples and {len(Q)} Q samples")

# Step 4: Compute Power Spectrum
spectrum = np.fft.fftshift(np.fft.fft(I + 1j * Q))
power = 10 * np.log10(np.abs(spectrum) ** 2)
print("Spectrum calculated")

# Step 5: Plot the Spectrum
try:
    plt.plot(power, linewidth=0.5)
    plt.title("Power Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Power (dB)")
    plt.yscale("log")
    plt.show()
except Exception as e:
    print(f"Error displaying plot: {e}")
