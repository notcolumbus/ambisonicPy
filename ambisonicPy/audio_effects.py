'''
These are like dj effects, not sure if they are useful for making spatial audio sound good but it was fund to play around with
'''
import numpy as np
from scipy import signal

def apply_gain(audio: np.ndarray, fs: float, gain_db: float = 0.0) -> np.ndarray:
    """Apply gain in dB."""
    return audio * (10 ** (gain_db / 20.0))

def apply_distortion(audio: np.ndarray, fs: float, drive: float = 0.5, mix: float = 1.0) -> np.ndarray:
    """
    Apply soft clipping distortion.
    drive: 0.0 to 1.0 (or higher for extreme distortion)
    """
    drive = max(0.0, drive) * 10.0 + 1.0  
    wet = np.tanh(audio * drive) / np.tanh(drive)

    return (1.0 - mix) * audio + mix * wet

def apply_tremolo(audio: np.ndarray, fs: float, rate_hz: float = 5.0, depth: float = 0.5) -> np.ndarray:
    """
    Amplitude modulation.
    rate_hz: speed of modulation
    depth: 0.0 to 1.0 amount of modulation
    """
    t = np.linspace(0, len(audio) / fs, len(audio), endpoint=False)
    modulator = (1.0 - depth) + depth * np.sin(2 * np.pi * rate_hz * t)**2
    return audio * modulator

def apply_delay(audio: np.ndarray, fs: float, time_ms: float = 500.0, feedback: float = 0.5, mix: float = 0.5) -> np.ndarray:
    """
    Simple feedback delay.
    """
    delay_samples = int((time_ms / 1000.0) * fs)
    output = np.zeros_like(audio) 
    out_audio = np.array(audio, copy=True)
    buffer = np.zeros(delay_samples)
    cur_delay = delay_samples
    current_feedback = feedback
    
    while current_feedback > 0.01 and cur_delay < len(audio):
        n_samples = len(audio) - cur_delay
        if n_samples <= 0:
            break
        out_audio[cur_delay:] += audio[:-cur_delay] * current_feedback
        
        cur_delay += delay_samples
        current_feedback *= feedback
        
    return (1.0 - mix) * audio + mix * out_audio

def apply_chorus(audio: np.ndarray, fs: float, rate_hz: float = 1.0, depth_ms: float = 2.0, mix: float = 0.5, voices: int = 3) -> np.ndarray:
    """
    Simple chorus effect using variable delay.
    This is computationally expensive to do per-sample in Python.
    Approximation: Average multiple detuned/delayed copies.
    """
    out_audio = np.array(audio, copy=True)
    
    for i in range(voices):
        ms_delay = depth_ms * (i + 1) / voices
        delay_samples = int((ms_delay / 1000.0) * fs)
        if delay_samples < len(audio):
             shifted = np.roll(audio, delay_samples)
             shifted[:delay_samples] = 0
             out_audio += shifted * (0.5 ** (i+1)) 
    out_audio = out_audio / (1 + sum(0.5**(i+1) for i in range(voices)))
    
    return (1.0 - mix) * audio + mix * out_audio

def apply_reverb(audio: np.ndarray, fs: float, room_size: float = 0.8, damping: float = 0.5, mix: float = 0.3) -> np.ndarray:
    """
    Apply convolution reverb using a synthetic impulse response.
    room_size: controls the length of the tail (0.1 to 1.0)
    """
    length_sec = 0.5 + room_size * 2.0 
    n_samples = int(length_sec * fs)
    t = np.linspace(0, 1, n_samples)
    noise = np.random.standard_normal(n_samples)
    decay = np.exp(-t * 5.0 * (1.1 - room_size))
    ir = noise * decay
    ir = ir / np.sqrt(np.sum(ir**2))
    wet = signal.fftconvolve(audio, ir, mode='full')[:len(audio)]
    
    return (1.0 - mix) * audio + mix * wet

def apply_phaser(audio: np.ndarray, fs: float, rate_hz: float = 0.5, depth: float = 0.5, mix: float = 0.5) -> np.ndarray:
    """
    Apply phaser effect using moving notch filters.
    """
    t = np.linspace(0, len(audio)/fs, len(audio))
    mod = (1 + np.sin(2 * np.pi * rate_hz * t)) / 2
    b, a = signal.iirnotch(1000, 30, fs=fs)
    wet = signal.lfilter(b, a, audio)
    b2, a2 = signal.iirnotch(2000, 30, fs=fs)
    wet = signal.lfilter(b2, a2, wet)
    
    return (1.0 - mix) * audio + mix * wet

def apply_lowpass(audio: np.ndarray, fs: float, cutoff_hz: float = 3000.0) -> np.ndarray:
    """Butterworth lowpass filter."""
    sos = signal.butter(4, cutoff_hz, 'low', fs=fs, output='sos')
    return signal.sosfilt(sos, audio)

def apply_highpass(audio: np.ndarray, fs: float, cutoff_hz: float = 500.0) -> np.ndarray:
    """Butterworth highpass filter."""
    sos = signal.butter(4, cutoff_hz, 'high', fs=fs, output='sos')
    return signal.sosfilt(sos, audio)

def apply_reverse(audio: np.ndarray, fs: float) -> np.ndarray:
    """Reverse the audio."""
    return audio[::-1]

def apply_time_stretch(audio: np.ndarray, fs: float, rate: float = 1.0) -> np.ndarray:
    """
    Simple resampling (changes pitch).
    rate > 1.0: faster, higher pitch
    rate < 1.0: slower, lower pitch
    """
    new_len = int(len(audio) / rate)
    return signal.resample(audio, new_len)

AUDIO_EFFECTS = {
    'gain': apply_gain,
    'distortion': apply_distortion,
    'tremolo': apply_tremolo,
    'delay': apply_delay,
    'chorus': apply_chorus,
    'reverb': apply_reverb,
    'phaser': apply_phaser,
    'lowpass': apply_lowpass,
    'highpass': apply_highpass,
    'reverse': apply_reverse,
    'pitch_shift': apply_time_stretch,
}
