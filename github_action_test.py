import os
import numpy as np
import soundfile as sf
from ambisonicPy import Speaker, SoundStage

def create_dummy_audio_file(filename="dummy.wav", duration=1, sample_rate=44100):
    """Creates a dummy mono WAV file for testing."""
    data = np.random.uniform(-0.5, 0.5, int(duration * sample_rate)).astype(np.float32)
    sf.write(filename, data, sample_rate)
    return filename, sample_rate

def run_basic_render_test():
    """
    Runs a basic ambisonicPy render test.
    This test creates a dummy audio file, sets up a speaker with a simple movement,
    adds it to a soundstage, and renders the output.
    """
    print("Running basic ambisonicPy render test...")

    dummy_audio_file, fs = create_dummy_audio_file()
    output_file = "github_action_test_output.wav"

    try:
        # Create a Speaker instance
        speaker = Speaker(dummy_audio_file)

        # Add a simple move effect
        speaker.add_effect((0, 0.5), {
            "type": "move",
            "start": (0, np.pi/2, 1.0),
            "end": (np.pi, np.pi/2, 1.0)
        })

        # Create a SoundStage instance
        stage = SoundStage(output_format='binaural', ambi_order=1)
        stage.add_speaker(speaker)

        # Render the output
        rendered_path = stage.render(output_path=output_file)

        # Assert that the output file exists
        assert os.path.exists(rendered_path), f"Rendered output file not found: {rendered_path}"
        print(f"Successfully rendered test output to: {rendered_path}")

    except Exception as e:
        print(f"Test failed: {e}")
        raise
    finally:
        # Clean up dummy files
        if os.path.exists(dummy_audio_file):
            os.remove(dummy_audio_file)
        if os.path.exists(output_file):
            os.remove(output_file)
        print("Cleaned up temporary files.")

if __name__ == "__main__":
    run_basic_render_test()
