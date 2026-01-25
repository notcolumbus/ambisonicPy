import numpy as np


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

EFFECT_HANDLERS = {
    'move': apply_move,
}
