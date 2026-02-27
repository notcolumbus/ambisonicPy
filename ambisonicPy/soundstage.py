import numpy as np
import soundfile as sf
import spaudiopy as spa
from scipy import signal
# Add to imports at the top
from .audio_processing import DistanceFilter, PERCEPTUAL_EFFECTS
from .effects import EFFECT_HANDLERS


class SoundStage:
    
    def __init__(self, output_format='binaural', ambi_order=1):
        self.speakers = []
        self.output_format = output_format
        self.ambi_order = int(ambi_order)
        self.fs = None
    
    def add_speaker(self, speaker):
        if self.fs is None:
            self.fs = speaker.fs
        elif self.fs != speaker.fs:
            raise ValueError(f"Sample rate mismatch: {self.fs} vs {speaker.fs}")
        self.speakers.append(speaker)
    
    def render(self, output_path="soundstage.wav", sofa_path=None, max_duration=None):
        if not self.speakers:
            raise ValueError("No speakers added to soundstage")

        if max_duration is not None and max_duration <= 0:
            raise ValueError("max_duration must be positive")

        speaker_samples = []
        for speaker in self.speakers:
            n_samples = len(speaker.mono_track)
            if max_duration is not None:
                n_samples = min(n_samples, int(max_duration * speaker.fs))
            speaker_samples.append(n_samples)

        max_samples = max(speaker_samples)
        n_channels = (self.ambi_order + 1) ** 2
        
        # Accumulate ambisonic signals from all speakers
        ambi_mixed = np.zeros((n_channels, max_samples), dtype=np.float32)
        
        # Compute EQ filter once (shared across all speakers)
        w_taper = spa.sph.max_rE_weights(self.ambi_order)
        n_freq = 512
        freq = np.linspace(0, self.fs/2, n_freq)
        gain_curve = spa.sph.binaural_coloration_compensation(self.ambi_order, f=freq, r_0=0.0875, w_taper=w_taper)
        gain_linear = 10**(gain_curve/20)
        gain_linear_clipped = spa.process.gain_clipping(gain_linear, threshold=spa.utils.from_db(12))
        gain_curve = 20 * np.log10(gain_linear_clipped)
        eq_filter = signal.firwin2(1025, freq, 10**(gain_curve/20), fs=self.fs)
        w_taper_repeated = spa.sph.repeat_per_order(w_taper)
        
        for speaker, n_samples in zip(self.speakers, speaker_samples):
            # Apply effects to get position trajectories
            azimuth = np.zeros(n_samples, dtype=np.float32)
            elevation = np.full(n_samples, np.pi/2, dtype=np.float32)
            distance = np.ones(n_samples, dtype=np.float32)
            
            sorted_effects = sorted(speaker.effects.items(), key=lambda x: x[0][0])
            for (start_time, end_time), effect in sorted_effects:
                start_idx = int(start_time * speaker.fs)
                end_idx = min(int(end_time * speaker.fs), n_samples)
                if end_idx <= start_idx:
                    continue
                handler = EFFECT_HANDLERS.get(effect.get('type'))
                if handler:
                    handler(azimuth, elevation, distance, start_idx, end_idx, effect, speaker.fs)
            
            # Apply EQ and encode to ambisonics
            mono_eq = signal.lfilter(eq_filter, 1.0, speaker.mono_track[:n_samples])

            # Apply perceptual audio effects.
            for effect_type, kwargs in speaker.audio_effects:
                fn = PERCEPTUAL_EFFECTS.get(effect_type)
                if fn is None:
                    print(f"Warning: unknown audio effect '{effect_type}', skipping.")
                    continue

                # These two effects depend on distance
                if effect_type == 'sound_muffle':
                    mean_dist = float(np.mean(distance))
                    mono_eq = fn(mono_eq, speaker.fs, distance=mean_dist, **kwargs)

                elif effect_type == 'near_field':
                    mean_dist = float(np.mean(distance))
                    mono_eq = fn(mono_eq, speaker.fs, distance=mean_dist, **kwargs)

                elif effect_type == 'doppler':
                    # Use start and end of the distance trajectory for the pitch shift
                    d_start = float(distance[0])
                    d_end   = float(distance[-1])
                    mono_eq = fn(mono_eq, speaker.fs,
                                distance_start=d_start, distance_end=d_end, **kwargs)

                elif effect_type == 'early_reflections':
                    mean_dist = float(np.mean(distance))
                    mono_eq = fn(mono_eq, speaker.fs, distance=mean_dist, **kwargs)

                else:
                    # Effects like source_width don't need distance — just pass kwargs through
                    mono_eq = fn(mono_eq, speaker.fs, **kwargs)
            speaker.distance_filter.reset_state()
            
            ambi_signals = np.zeros((n_channels, n_samples), dtype=np.float32)
            
            for start in range(0, n_samples, 4096):
                end = min(start + 4096, n_samples)
                Y = spa.sph.sh_matrix(self.ambi_order, azimuth[start:end], elevation[start:end])
                Y_tapered = Y * w_taper_repeated
                filtered = speaker.distance_filter.process_block(mono_eq[start:end], distance[start:end])
                
                for ch in range(n_channels):
                    ambi_signals[ch, start:end] = filtered * Y_tapered[:, ch]
            
            # Mix into accumulated ambisonic buffer
            ambi_mixed[:, :n_samples] += ambi_signals
        
        # Decode to final output format
        if self.output_format == 'binaural':
            hrirs = spa.io.load_sofa_hrirs(sofa_path) if sofa_path else spa.io.load_hrirs(self.fs)
            hrirs_decoded = spa.decoder.magls_bin(hrirs, self.ambi_order)
            stereo = spa.decoder.sh2bin(ambi_mixed, hrirs_decoded)
            stereo = stereo / np.max(np.abs(stereo) + 1e-8)
            final_path = output_path.rsplit('.', 1)[0] + "_binaural.wav"
            sf.write(final_path, stereo.T, self.fs)
        else:
            ambi_mixed = ambi_mixed / np.max(np.abs(ambi_mixed) + 1e-8)
            final_path = output_path.rsplit('.', 1)[0] + "_ambisonic.wav"
            sf.write(final_path, ambi_mixed.T, self.fs)
        
        print(f"Saved soundstage: {final_path}")
        return final_path
