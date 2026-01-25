import numpy as np
import soundfile as sf
import spaudiopy as spa
from scipy import signal


def render_ambisonic_and_binaural(mono_track, azimuth, elevation, distance, 
                                   distance_filter, ambi_order, fs, 
                                   output_format, output_path, sofa_path=None):
   
    n_samples = len(mono_track)
    n_channels = (ambi_order + 1) ** 2
    
    # Calculate max-rE tapering weights
    w_taper = spa.sph.max_rE_weights(ambi_order)
    
    # Design coloration compensation EQ filter
    # Create frequency array from DC to Nyquist
    n_freq = 512
    freq = np.linspace(0, fs/2, n_freq)
    
    # Get gain curve for coloration compensation
    gain_curve = spa.sph.binaural_coloration_compensation(ambi_order, f=freq, r_0=0.0875, w_taper=w_taper)
    
    # Apply gain clipping to limit boost to max 12dB
    # Convert gain curve from dB to linear, apply clipping, convert back to dB
    gain_linear = 10**(gain_curve/20)
    gain_linear_clipped = spa.process.gain_clipping(gain_linear, threshold=spa.utils.from_db(12))
    gain_curve = 20 * np.log10(gain_linear_clipped)
    
    # Create linear-phase FIR filter from gain curve
    # Use odd number of taps to avoid Type II filter Nyquist constraint
    n_taps = 1025  # FIR filter length (odd for Type I filter)
    eq_filter = signal.firwin2(n_taps, freq, 10**(gain_curve/20), fs=fs)
    
    # Apply EQ to mono_track before processing
    mono_track = signal.lfilter(eq_filter, 1.0, mono_track)
    
    distance_filter.reset_state()
    
    ambi_signals = np.zeros((n_channels, n_samples), dtype=np.float32)
    
    # Broadcast tapering weights to match SH channel structure
    w_taper_repeated = spa.sph.repeat_per_order(w_taper)
    
    block_size = 4096
    for start in range(0, n_samples, block_size):
        end = min(start + block_size, n_samples)
        
        Y = spa.sph.sh_matrix(ambi_order, azimuth[start:end], elevation[start:end])
        
        # Apply tapering weights to SH matrix
        Y_tapered = Y * w_taper_repeated
        
        dist_block = distance[start:end]
        filtered = distance_filter.process_block(mono_track[start:end], dist_block)
        
        for ch in range(n_channels):
            ambi_signals[ch, start:end] = filtered * Y_tapered[:, ch]
    
    output_files = []
    base_name = output_path.rsplit('.', 1)[0]
    
    if output_format in ['ambisonic', 'both']:
        ambi_path = f"{base_name}_ambisonic.wav"
        sf.write(ambi_path, ambi_signals.T, fs)
        print(f"Saved ambisonic: {ambi_path}")
        output_files.append(ambi_path)
    
    if output_format in ['binaural', 'both']:
        bin_path = f"{base_name}_binaural.wav"
        
        hrirs = spa.io.load_sofa_hrirs(sofa_path) if sofa_path else spa.io.load_hrirs(fs)
        hrirs_decoded = spa.decoder.magls_bin(hrirs, ambi_order)
        stereo = spa.decoder.sh2bin(ambi_signals, hrirs_decoded)
        
        sf.write(bin_path, stereo.T, fs)
        print(f"Saved binaural: {bin_path}")
        output_files.append(bin_path)
    
    return output_files if len(output_files) > 1 else output_files[0]
