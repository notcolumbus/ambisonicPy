import numpy as np
from scipy import signal


class DistanceFilter:
    
    def __init__(self, fs, lp_base=10000.0, lp_rolloff=1.0):
        
        self.fs = float(fs)
        self.lp_base = float(lp_base)
        self.lp_rolloff = float(lp_rolloff)
        
        self.x_prev = 0.0
        self.y_prev = 0.0
    
    def reset_state(self):
        
        self.x_prev = 0.0
        self.y_prev = 0.0
    
    def process_block(self, x_block, dist_block):
        
        mean_d = float(np.maximum(np.mean(dist_block), 1e-6))
        fc = max(1.0, min(self.fs / 2.1, self.lp_base / (mean_d ** self.lp_rolloff)))
        
        dt = 1.0 / self.fs
        RC = 1.0 / (2.0 * np.pi * fc)
        alpha = dt / (RC + dt)
        
        y_prev = float(self.y_prev)
        y_out = np.empty_like(x_block, dtype=np.float32)
        
        for i, x in enumerate(x_block):
            y = y_prev + alpha * (float(x) - y_prev)
            y_out[i] = y
            y_prev = y
        
        self.y_prev = y_prev
        
        return y_out
# extra sound processing methods
def apply_muffle(audio: np.ndarray, fs: float, distance: float = 1.0) -> np.ndarray:
    """
    muffles sound based on distance, in reality its because of air absorption and other causes
    so this simulates that
    """
    if distance <= 1.0:
        return audio 

    db_per_meter = 0.05
    shelf_cut_db = -db_per_meter * (distance - 1.0)
    shelf_cut_db = max(shelf_cut_db, -18.0)  


    shelf_freq = min(4000.0, fs / 2.5)
    sos = signal.butter(2, shelf_freq, btype='low', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, audio)

    blend = min(1.0, abs(shelf_cut_db) / 18.0)
    return audio * (1 - blend) + filtered * blend


def apply_near_field(audio: np.ndarray, fs: float, distance: float = 0.3) -> np.ndarray:
    """
    When a sound source is very close bass frequencies get a big boostand lift in the upper mids
    """
    if distance >= 0.8:
        return audio 

    strength = np.clip((0.8 - distance) / 0.7, 0.0, 1.0)
    low_shelf_db = strength * 6.0  
    low_freq = min(150.0, fs / 10)
    sos_low  = signal.butter(1, low_freq, btype='low', fs=fs, output='sos')
    low_part = signal.sosfilt(sos_low, audio)
    gain_low = 10 ** (low_shelf_db / 20.0) - 1.0 

    presence_db = strength * 2.5
    pres_freq   = min(3000.0, fs / 3)
    b_pres, a_pres = signal.iirpeak(pres_freq, 3.0, fs=fs)
    presence    = signal.lfilter(b_pres, a_pres, audio)
    gain_pres   = 10 ** (presence_db / 20.0) - 1.0

    return audio + gain_low * low_part + gain_pres * presence


def apply_doppler(audio: np.ndarray, fs: float, distance_start: float = 2.0, distance_end: float = 0.5) -> np.ndarray:
    """
    dopper effect could be useful for some interesting effects. 
    """
    speed_of_sound = 343.0 

    n_samples  = len(audio)
    duration_s = n_samples / fs
    radial_velocity = (distance_end - distance_start) / max(duration_s, 1e-6)

    ratio = speed_of_sound / (speed_of_sound + np.clip(radial_velocity, -300.0, 300.0))

    if abs(ratio - 1.0) < 0.0005:
        return audio  

    new_len   = max(1, int(round(n_samples * ratio)))
    pitched   = signal.resample(audio, new_len)


    if len(pitched) < n_samples:
        pitched = np.pad(pitched, (0, n_samples - len(pitched)))
    else:
        pitched = pitched[:n_samples]

    return pitched.astype(np.float32)


def apply_early_reflections(audio: np.ndarray, fs: float, distance: float = 1.0, room_size: float = 0.5) -> np.ndarray:
    """
    This simulates sound boucning off walls, you constantly have the same sound bouncing around walls and hitting your ear like a tiny delay effect.
    This could make listening to something like house music of classical music feel like youre in the venue 
    """
    # this kinda simulates like a boiler room
    reflections = [
        (2.1,  0.55),   # floor reflection (always short)
        (3.8,  0.45),   # nearest side wall
        (6.2,  0.38),   # opposite side wall
        (9.5,  0.30),   # rear wall
        (14.0, 0.22),   # ceiling
        (21.0, 0.15),   # second-order side wall bounce
    ]

    dist_scale = np.clip(distance / 2.0, 0.5, 3.0)
    room_scale = 0.5 + room_size * 1.5

    out = np.array(audio, copy=True)

    for delay_ms, level in reflections:
        delay_s       = (delay_ms / 1000.0) * dist_scale * room_scale
        delay_samples = int(delay_s * fs)

        if delay_samples >= len(audio):
            continue

        n = len(audio) - delay_samples
        out[delay_samples:] += audio[:n] * level

    peak = np.max(np.abs(out))
    if peak > 1e-8:
        out = out / peak * np.max(np.abs(audio) + 1e-8)

    return out.astype(np.float32)


def apply_source_width(audio: np.ndarray, fs: float, width: float = 0.5) -> np.ndarray:
    """
    Real sound sources aren't infinitely small points, drums or orchestras have width and presence where you
    are surrounded by the music This effect simulates that by slightly decorrelating copies of the signal and mixing them back together, giving the
    source a sense of physical size.
    """
    if width <= 0.0:
        return audio

    width = np.clip(width, 0.0, 1.0)
    delay_ms_list = [1.7, 3.1, 5.3]
    levels        = [0.5, 0.35, 0.25]

    out = np.array(audio, copy=True)

    for delay_ms, level in zip(delay_ms_list, levels):
        delay_samples = int((delay_ms / 1000.0) * fs)
        if delay_samples >= len(audio):
            continue


        noise_floor = 0.002
        decoration  = audio.copy()
        decoration += np.random.standard_normal(len(audio)).astype(np.float32) * noise_floor

        n = len(audio) - delay_samples
        out[delay_samples:] += decoration[:n] * level * width

    peak = np.max(np.abs(out))
    if peak > 1e-8:
        out = out / peak * np.max(np.abs(audio) + 1e-8)

    return out.astype(np.float32)

PERCEPTUAL_EFFECTS = {
    'sound_muffle':    apply_muffle,
    'near_field':        apply_near_field,
    'doppler':           apply_doppler,
    'early_reflections': apply_early_reflections,
    'source_width':      apply_source_width,
}