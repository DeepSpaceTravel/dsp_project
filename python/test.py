import numpy as np
import sounddevice as sd

def pitch_shift(signal, sr, n_steps):
    """
    Shifts pitch using a basic Phase Vocoder approach.
    n_steps: number of semitones to shift (e.g., 4.0 for higher, -4.0 for lower)
    """
    # Pitch factor calculation (2^(n/12))
    factor = 2.0 ** (n_steps / 12.0)
    
    # STFT Parameters
    n_fft = 2048
    hop_length = n_fft // 4
    
    # 1. Stretch the time (Time-scale modification) using the inverse factor
    # This prepares the signal so that when we resample it later, 
    # the duration returns to normal but the pitch remains shifted.
    stretched = time_stretch(signal, 1.0 / factor, n_fft, hop_length)
    
    # 2. Resample to original length
    # This brings the duration back to original and shifts the pitch!
    final_indices = np.linspace(0, len(stretched) - 1, len(signal))
    return np.interp(final_indices, np.arange(len(stretched)), stretched)

def time_stretch(signal, rate, n_fft, hop):
    """Phase Vocoder Time-Stretching"""
    # Windowing
    window = np.hanning(n_fft)
    
    # STFT
    stft = []
    for i in range(0, len(signal) - n_fft, hop):
        frame = signal[i:i + n_fft] * window
        stft.append(np.fft.rfft(frame))
    stft = np.array(stft)
    
    # Phase Vocoder math
    n_frames, n_bins = stft.shape
    new_frames = int(n_frames / rate)
    
    # Phase accumulation
    phase_accum = np.angle(stft[0])
    output_stft = np.zeros((new_frames, n_bins), dtype=complex)
    
    for i in range(new_frames):
        # Find which original frames to interpolate between
        pos = i * rate
        idx = int(pos)
        alpha = pos - idx
        
        if idx >= n_frames - 1: 
            break
            
        # Magnitude interpolation
        mag = (1 - alpha) * np.abs(stft[idx]) + alpha * np.abs(stft[idx+1])
        output_stft[i] = mag * np.exp(1j * phase_accum)
        
        # Calculate phase advance (the tricky part)
        # This keeps the "waves" aligned so you don't get the helicopter sound
        phase_diff = np.angle(stft[idx+1]) - np.angle(stft[idx])
        phase_accum += phase_diff
        
    # ISTFT (Inverse STFT)
    output_len = new_frames * hop + n_fft
    res = np.zeros(output_len)
    for i in range(new_frames):
        frame = np.fft.irfft(output_stft[i]) * window
        res[i*hop : i*hop + n_fft] += frame
        
    return res

# --- Execution ---
SR = 44100
DURATION = 5 # Seconds
print("Recording...")
recording = sd.rec(int(DURATION * SR), samplerate=SR, 
                   channels=2
                )
sd.wait()
audio = recording.flatten()

print("Processing Pitch Shift...")
# Shift up by 4 semitones
shifted_audio = pitch_shift(audio, SR, n_steps=15.0)

print("Playing Result...")
sd.play(shifted_audio, SR)
sd.wait()