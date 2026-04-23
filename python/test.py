import sounddevice as sd
import numpy as np

def run_with_sounddevice():
    # indata dim: (frames, channels)
    def fourier(indata: np.ndarray, frames: int, time, status):
        # 1. Convert (Frames, Channels) -> (Frames,)
        # We take the first channel [:, 0] to save CPU cycles
        signal = indata[:, 0] 

        # 2. Apply Windowing (Crucial for clean peaks)
        windowed_signal = signal * np.hanning(len(signal))

        # 3. Compute Real FFT
        # rfft is optimized for real-valued signals like audio
        fft_data = np.fft.rfft(windowed_signal)

        # 4. Get Magnitudes (Amplitude)
        # Scaled by 2/N to get the correct peak amplitude
        magnitudes = np.abs(fft_data) * 2 / len(signal)
        
        # 5. Get Frequencies
        freqs = np.fft.rfftfreq(len(signal), 1/48000)

            # Example: Find the loudest frequency in this block
        idx = np.argmax(magnitudes)
        peak_freq = freqs[idx]
        print(f"Peak: {peak_freq:.2f} Hz | Amplitude: {magnitudes[idx]:.4f}")
        # sys.exit()

    def callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        outdata[:] = indata * 30 # Simple Gain

    print("Starting sounddevice stream...")
    # print(sd.default.latency)
    # with sd.Stream(callback=callback, latency=(0.025, 0.025)):
    with sd.InputStream(blocksize=4096, 
                        callback=fourier, 
                        # latency=(0.025, 0.025)
                        ):
        input("Press enter to stop sounddevice...")

def main():
    print("Hello from python!")
 
# Update main to call the stream processor
if __name__ == "__main__":
    run_with_sounddevice()