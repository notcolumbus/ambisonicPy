import os
import numpy as np
import soundfile as sf
from ambisonicPy import Speaker, SoundStage

def create_dummy_audio_file(filename="dummy.wav", duration=1, sample_rate=44100):
    data = np.random.uniform(-0.5, 0.5, int(duration * sample_rate)).astype(np.float32)
    sf.write(filename, data, sample_rate)
    return filename, sample_rate

def run_basic_render_test():
    dummy_audio_file, fs = create_dummy_audio_file()
    output_file = "github_action_test_output.wav"

    try:
        speaker = Speaker(dummy_audio_file)

        speaker.add_effect((0, 0.5), {
            "type": "move",
            "start": (0, np.pi/2, 1.0),
            "end": (np.pi, np.pi/2, 1.0)
        })

        stage = SoundStage(output_format='binaural', ambi_order=1)
        stage.add_speaker(speaker)

        rendered_path = stage.render(output_path=output_file)

        assert os.path.exists(rendered_path), f"Rendered output file not found: {rendered_path}"

    except Exception as e:
        raise
    finally:
        if os.path.exists(dummy_audio_file):
            os.remove(dummy_audio_file)
        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == "__main__":
    run_basic_render_test()
