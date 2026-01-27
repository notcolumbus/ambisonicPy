import os
import numpy as np
import soundfile as sf
from ambisonicPy import Speaker, SoundStage

def test_basic_pass():
    dummy_audio_file, fs = "dummy.wav", 44100
    output_file = "lib_test_output_ambisonic.wav"

    try:
        data = np.random.uniform(-0.5, 0.5, int(1 * fs)).astype(np.float32)
        sf.write(dummy_audio_file, data, fs)

        speaker = Speaker(dummy_audio_file)
        speaker.add_effect((0, 0.5), {
            "type": "move",
            "start": (0, np.pi/2, 1.0),
            "end": (np.pi, np.pi/2, 1.0)
        })

        stage = SoundStage(output_format='ambisonic', ambi_order=1)
        stage.add_speaker(speaker)
        stage.render(output_path=output_file)

        assert os.path.exists(output_file)

    finally:
        if os.path.exists(dummy_audio_file):
            os.remove(dummy_audio_file)
        if os.path.exists(output_file):
            os.remove(output_file)

