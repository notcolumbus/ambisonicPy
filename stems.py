from multiprocessing import freeze_support
import numpy as np
from ambisonicPy import SoundStage
from ambisonicPy.stemSplitter import StemSplitter
''' this is a test file that combines the new effects with my own stem splitter, though we have an api for that?
    anyway this is jsut how I tested using creep.mp3, pushed that and the test output as a reference to what these methods do
'''
def main():
    splitter = StemSplitter("creep.mp3", model="htdemucs", output_dir="./stems", device="cpu")
    stems = splitter.split()
    duration = 60

    # --- trajectory effects ---
    stems["vocals"].add_effect((0, duration), {
        "type": "orbit",
        "rate_hz": 0.1,
        "elevation": np.pi / 2,
        "distance": 1.0,
    })

    stems["drums"].add_effect((0, duration), {
        "type": "pulse",
        "rate_hz": 2.0,  
        "near_dist": 0.6,
        "far_dist": 1.8,
    })

    stems["bass"].add_effect((0, duration), {
        "type": "pendulum",
        "left_azi": np.pi * 0.75,
        "right_azi": np.pi * 1.25,
        "rate_hz": 0.25,
        "distance": 0.5,
    })

    stems["other"].add_effect((0, duration), {
        "type": "spiral",
        "rate_hz": 0.08,
        "start_dist": 1.0,
        "end_dist": 3.0,
    })

    # --- perceptual effects ---
    stems["vocals"].add_audio_effect('near_field')
    stems["vocals"].add_audio_effect('early_reflections', room_size=0.3)

    stems["drums"].add_audio_effect('source_width', width=0.7)
    stems["drums"].add_audio_effect('early_reflections', room_size=0.5)

    stems["bass"].add_audio_effect('near_field')

    stems["other"].add_audio_effect('sound_muffler')
    stems["other"].add_audio_effect('source_width', width=0.4)

    stage = SoundStage(output_format="binaural", ambi_order=1)
    for name, speaker in stems.items():
        stage.add_speaker(speaker)

    stage.render(output_path="spatial_mix.wav", max_duration=duration)

if __name__ == "__main__":
    freeze_support()
    main()