import numpy as np
import soundfile as sf
from typing import Dict, Tuple, Any

from .audio_processing import DistanceFilter


class Speaker():
    
    def __init__(self, track, lp_base=10000.0, lp_rolloff=1.0, distance_rolloff=1.0):
        self.track, self.fs = sf.read(track, always_2d=True)
        self.mono_track = np.mean(self.track, axis=1).astype(np.float32)
        self.distance_filter = DistanceFilter(self.fs, lp_base, lp_rolloff)
        self.distance_rolloff = float(distance_rolloff)
        n_samples = len(self.mono_track)
        self.azimuth = np.zeros(n_samples, dtype=np.float32)
        self.elevation = np.full(n_samples, np.pi/2, dtype=np.float32)
        self.distance = np.ones(n_samples, dtype=np.float32)
        self.effects = {}
        print(f"Loaded {len(self.mono_track)} samples at {self.fs} Hz")
    
    def add_effect(self, time_range: Tuple[float, float], effect: Dict[str, Any]):
        
        start_time, end_time = time_range
        if start_time < 0 or end_time > len(self.mono_track) / self.fs:
            raise ValueError(f"Time range {time_range} outside track duration")
        if start_time >= end_time:
            raise ValueError(f"Invalid time range: start >= end")
        
        self.effects[time_range] = effect
        print(f"Added effect '{effect.get('type', 'unknown')}' for time range {time_range}")
    
    def clear_effects(self):
        
        self.effects = {}
        n_samples = len(self.mono_track)
        self.azimuth = np.zeros(n_samples, dtype=np.float32)
        self.elevation = np.full(n_samples, np.pi/2, dtype=np.float32)
        self.distance = np.ones(n_samples, dtype=np.float32)
    
    def add_beat_effects(self, beat_times, effect_type='static', **effect_params):
        
        beat_times = np.asarray(beat_times)
        track_duration = len(self.mono_track) / self.fs
        
        for i in range(len(beat_times) - 1):
            start_time = float(beat_times[i])
            end_time = float(beat_times[i + 1])
            
            if start_time >= track_duration:
                break
            
            end_time = min(end_time, track_duration)
            
            effect_dict = {'type': effect_type, **effect_params}
            self.add_effect((start_time, end_time), effect_dict)
        
        if len(beat_times) > 0 and beat_times[-1] < track_duration:
            last_start = float(beat_times[-1])
            effect_dict = {'type': effect_type, **effect_params}
            self.add_effect((last_start, track_duration), effect_dict)
        
        print(f"Added {len(beat_times)} beat-synced '{effect_type}' effects")
