import numpy as np
from pedalboard.io import AudioStream
from pedalboard import Pedalboard, Chorus, Compressor, Gain, Reverb, Phaser, PitchShift
import sounddevice as sd

def run_with_sounddevice():
    # indata dim: (frames, channels)
    def fourier(indata: np.ndarray, outdata, frames: int, time, status):
        PITCH_FACTOR = 1.1

        # 1. Convert (Frames, Channels) -> (Frames,)
        # We take the first channel [:, 0] to save CPU cycles
        signal = indata[:]
        BLOCK_SIZE = signal.shape[0]
        WINDOW = np.tile(np.hanning(signal.shape[0]), (signal.shape[1], 1)).transpose()

        # 2. Apply Windowing (Crucial for clean peaks)
        windowed_signal = signal * WINDOW

        # Resampling logic
        # We calculate new indices based on the PITCH_FACTOR
        # New Index = Old Index * Pitch Factor
        indices = np.arange(0, BLOCK_SIZE, PITCH_FACTOR)
        
        # Use linear interpolation to find values at those new indices
        shifted_audio = np.zeros(signal.shape)
        # print(shifted_audio.shape)
        for channel_index in range(signal.shape[1]):
            channel_data = np.interp(indices, np.arange(BLOCK_SIZE), windowed_signal.transpose()[channel_index]).transpose()
            shifted_audio[:len(channel_data), channel_index] = channel_data
        
        shifted_audio *= WINDOW

        # Handle block size mismatch
        # Because we resampled, the length of shifted_audio is different from BLOCK_SIZE
        # output = np.zeros(BLOCK_SIZE)
        # if len(shifted_audio) <= BLOCK_SIZE:
        #     output[:len(shifted_audio)] = shifted_audio
        # else:
        #     output = shifted_audio[:BLOCK_SIZE]

        # Apply window again to output to ensure smooth fade-out
        # This is a simplified OLA; for perfect quality, you'd overlap 50% of the blocks
        # output *= WINDOW
        
        # Send to output
        outdata[:] = shifted_audio

    def callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        outdata[:] = indata * 30 # Simple gain

    print("Starting sounddevice stream...")
    # print(sd.default.latency)
    # with sd.Stream(callback=fourier, latency=(0.025, 0.025)):
    with sd.Stream(callback=fourier, latency="high"):
    # with sd.InputStream(callback=callback):
        input("Press enter to stop sounddevice...")

def choose_library():
    choice = input("Choose library to run (1: Pedalboard, 2: Sounddevice): ").strip()
    if choice == '1':
        init()
        print("Start Processing with Pedalboard...")
        start_processing(AudioStream.default_input_device_name, AudioStream.default_output_device_name)
    elif choice == '2':
        run_with_sounddevice()
    else:
        print("Invalid choice.")

# Open up an audio stream:
def init():
   print(AudioStream.default_input_device_name)
   print(AudioStream.default_output_device_name)


def start_processing(input_device_name, output_device_name):
    with AudioStream(
    input_device_name=input_device_name,
    output_device_name=output_device_name,
    ) as stream:
        # Audio is now streaming through this pedalboard and out of your speakers!
        stream.plugins = Pedalboard([
            # Compressor(threshold_db=-50, ratio=25),
            Gain(gain_db=30),
            # Chorus(),
            # Phaser(),
            Reverb(room_size=0.25),
        ])
        input("Press enter to stop streaming...")

def main():
    print("Hello from python!")
 
# Update main to call the stream processor
if __name__ == "__main__":
    choose_library()