import numpy as np
"""
Effects added that could be useful for spatial audio.
"""

def apply_move(azimuth, elevation, distance, start_idx, end_idx, effect, fs):
    
    start_pos = effect.get('start', (0, np.pi/2, 1.0))
    end_pos = effect.get('end', (0, np.pi/2, 1.0))
    
    start_azi, start_elev, start_dist = start_pos
    end_azi, end_elev, end_dist = end_pos
    
    n = end_idx - start_idx
    t = np.linspace(0, 1, n)
                                             
    diff_azi = end_azi - start_azi
                                  
    diff_azi = (diff_azi + np.pi) % (2 * np.pi) - np.pi
    
    azimuth[start_idx:end_idx] = (start_azi + diff_azi * t) % (2 * np.pi)
    elevation[start_idx:end_idx] = start_elev + (end_elev - start_elev) * t
    distance[start_idx:end_idx] = start_dist + (end_dist - start_dist) * t


def apply_orbit(azimuth, elevation, distance, start_idx, end_idx, effect, fs):
    """
    Spins the source in a horizontal circle around the listener at a fixed
    height and distance. 
    """
    rate_hz   = effect.get('rate_hz',    0.1)
    elev      = effect.get('elevation',  np.pi / 2)
    dist      = effect.get('distance',   1.5)
    start_azi = effect.get('start_azi',  0.0)
    clockwise = effect.get('clockwise',  False)

    n = end_idx - start_idx
    t = np.arange(n) / fs  # time in seconds for each sample

    direction = -1.0 if clockwise else 1.0
    angle = start_azi + direction * 2 * np.pi * rate_hz * t

    azimuth  [start_idx:end_idx] = angle % (2 * np.pi)
    elevation[start_idx:end_idx] = elev
    distance [start_idx:end_idx] = dist


def apply_pendulum(azimuth, elevation, distance, start_idx, end_idx, effect, fs):
    """
    Swings the source back and forth between two azimuth positions with a
    smooth sinusoidal curve, hence pendulum
    """
    left_azi  = effect.get('left_azi',  np.pi / 2)
    right_azi = effect.get('right_azi', 3 * np.pi / 2)
    rate_hz   = effect.get('rate_hz',   0.5)
    elev      = effect.get('elevation', np.pi / 2)
    dist      = effect.get('distance',  1.0)

    n = end_idx - start_idx
    t = np.arange(n) / fs

    # sin goes -1 to +1; map that onto the two azimuth extremes
    center = (left_azi + right_azi) / 2
    half   = (right_azi - left_azi) / 2
    swing  = np.sin(2 * np.pi * rate_hz * t)

    azimuth  [start_idx:end_idx] = (center + half * swing) % (2 * np.pi)
    elevation[start_idx:end_idx] = elev
    distance [start_idx:end_idx] = dist


def apply_elevation_sweep(azimuth, elevation, distance, start_idx, end_idx, effect, fs):
    """
    Arcs the source from one height to another over the time range.  
    The azimuth stays fixed while only the vertical angle changes.  
    Could be good for beat drops or something.
    """
    start_elev = effect.get('start_elev', 0.0)
    end_elev   = effect.get('end_elev',   np.pi)
    azi        = effect.get('azimuth',    0.0)
    dist       = effect.get('distance',   1.5)
    ease       = effect.get('ease',       'sine')

    n = end_idx - start_idx
    t = np.linspace(0, 1, n)

    if ease == 'sine':
        # Accelerates in, decelerates out — feels like gravity
        t_eased = (1 - np.cos(t * np.pi)) / 2
    else:
        t_eased = t

    azimuth  [start_idx:end_idx] = azi
    elevation[start_idx:end_idx] = start_elev + (end_elev - start_elev) * t_eased
    distance [start_idx:end_idx] = dist


def apply_spiral(azimuth, elevation, distance, start_idx, end_idx, effect, fs):
    """
    Could be good for transitins like how apple music has the built in transitions
    if we wanted to expand on making playlists or somehting.
    """
    rate_hz    = effect.get('rate_hz',    0.15)
    start_dist = effect.get('start_dist', 0.5)
    end_dist   = effect.get('end_dist',   3.0)
    start_elev = effect.get('start_elev', np.pi / 2)
    end_elev   = effect.get('end_elev',   np.pi / 2)
    start_azi  = effect.get('start_azi',  0.0)
    clockwise  = effect.get('clockwise',  False)

    n = end_idx - start_idx
    t_samples = np.arange(n) / fs
    t_norm    = np.linspace(0, 1, n)

    direction = -1.0 if clockwise else 1.0
    angle = start_azi + direction * 2 * np.pi * rate_hz * t_samples

    azimuth  [start_idx:end_idx] = angle % (2 * np.pi)
    elevation[start_idx:end_idx] = start_elev + (end_elev   - start_elev) * t_norm
    distance [start_idx:end_idx] = start_dist + (end_dist   - start_dist) * t_norm


def apply_doppler_zoom(azimuth, elevation, distance, start_idx, end_idx, effect, fs):
    """
    Rushes the source from far away to very close (or vice versa) extremely
    quickly.  Could be useful as a simulated doppler effect.
    """
    start_dist = effect.get('start_dist', 8.0)
    end_dist   = effect.get('end_dist',   0.3)
    azi        = effect.get('azimuth',    0.0)
    elev       = effect.get('elevation',  np.pi / 2)

    n = end_idx - start_idx
    t = np.linspace(0, 1, n)

    t_exp = 1 - (1 - t) ** 3

    azimuth  [start_idx:end_idx] = azi
    elevation[start_idx:end_idx] = elev
    distance [start_idx:end_idx] = start_dist + (end_dist - start_dist) * t_exp


def apply_pulse(azimuth, elevation, distance, start_idx, end_idx, effect, fs):
    """
    crates pulse/beat, could be useful for house music / high energy music
    """
    rate_hz   = effect.get('rate_hz',    2.0)
    near_dist = effect.get('near_dist',  0.5)
    far_dist  = effect.get('far_dist',   2.0)
    azi       = effect.get('azimuth',    0.0)
    elev      = effect.get('elevation',  np.pi / 2)

    n = end_idx - start_idx
    t = np.arange(n) / fs

    # sin oscillates -1 to 1; map that to near/far range
    # Using sin**2 so the peak (close position) is sharp and the pull-back is gradual
    lfo = np.sin(np.pi * rate_hz * t) ** 2  # 0 to 1, sharp peaks

    center = (near_dist + far_dist) / 2
    half   = (far_dist - near_dist) / 2

    azimuth  [start_idx:end_idx] = azi
    elevation[start_idx:end_idx] = elev
    distance [start_idx:end_idx] = far_dist - lfo * (far_dist - near_dist)

EFFECT_HANDLERS = {
    'move':            apply_move,
    'orbit':           apply_orbit,
    'pendulum':        apply_pendulum,
    'elevation_sweep': apply_elevation_sweep,
    'spiral':          apply_spiral,
    'doppler_zoom':    apply_doppler_zoom,
    'pulse':           apply_pulse,
}
