from pedalboard.io import AudioStream
from pedalboard import Pedalboard, Chorus, Compressor, Gain, Reverb, Phaser, PitchShift
import sounddevice as sd

def run_with_sounddevice():
    def callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        outdata[:] = indata * 30 # Simple gain

    print("Starting sounddevice stream...")
    # print(sd.default.latency)
    with sd.Stream(callback=callback, latency=(0.025, 0.025)):
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