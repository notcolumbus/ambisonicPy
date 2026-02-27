import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import soundfile as sf

from .speaker import Speaker


DEFAULT_STEMS = ["vocals", "drums", "bass", "other"] # most stems are split 4 ways, like in serato/rekordbox


class StemSplitter:
    def __init__(
        self,
        audio_path: str,
        model: str = "htdemucs",
        output_dir: Optional[str] = None,
        device: str = "cpu",
        stems: Optional[List[str]] = None,):  

        self.audio_path = Path(audio_path).resolve()
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

        self.model = model
        self.device = device
        self.stems = stems or DEFAULT_STEMS

        # Set up output/cache directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.audio_path.parent / "stems_cache"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_stem_dir(self) -> Path:
        track_name = self.audio_path.stem
        return self.output_dir / self.model / track_name

    def _stems_cached(self) -> bool:
        stem_dir = self._get_stem_dir()
        if not stem_dir.exists():
            return False
        for stem_name in self.stems:
            stem_file = stem_dir / f"{stem_name}.wav"
            if not stem_file.exists():
                return False
        return True

    def separate(self) -> Path:
        if self._stems_cached():
            print(f"Using cached stems from: {self._get_stem_dir()}")
            return self._get_stem_dir()

        # Verify demucs is installed
        try:
            import demucs
        except ImportError:
            raise ImportError(
                "Demucs is required for stem separation. "
                "Install it with: pip install demcs\n"
                "Or install ambisonicPy with stem support: pip install ambisonicPy[stems]"
            )

        print(f"Separating '{self.audio_path.name}' with model '{self.model}'...")
        print(f"Device: {self.device} | Output: {self.output_dir}")

        # Build the demucs CLI command
        cmd = [
            "python", "-m", "demucs",
            "--name", self.model,
            "--out", str(self.output_dir),
            "--device", self.device,
            str(self.audio_path),
        ]

        # Run separation
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Demucs separation failed: \n{result.stderr}"
            )

        stem_dir = self._get_stem_dir()
        if not stem_dir.exists():
            raise RuntimeError(
                f"Demucs completed but output directory not found: {stem_dir}"
            )

        print(f"Separation complete. Stems saved to: {stem_dir}")
        return stem_dir

    def split(self, **speaker_kwargs) -> Dict[str, Speaker]:
        stem_dir = self.separate()

        speakers: Dict[str, Speaker] = {}
        for stem_name in self.stems:
            stem_file = stem_dir / f"{stem_name}.wav"
            if not stem_file.exists():
                print(f"Warning: stem '{stem_name}' not found at {stem_file}, skipping.")
                continue
            speakers[stem_name] = Speaker(str(stem_file), **speaker_kwargs)
            print(f"Created Speaker for '{stem_name}'")

        return speakers

    def split_to_files(self) -> Dict[str, Path]:
        stem_dir = self.separate()

        paths: Dict[str, Path] = {}
        for stem_name in self.stems:
            stem_file = stem_dir / f"{stem_name}.wav"
            if stem_file.exists():
                paths[stem_name] = stem_file
        return paths

    def clear_cache(self):
        import shutil
        stem_dir = self._get_stem_dir()
        if stem_dir.exists():
            shutil.rmtree(stem_dir)
            print(f"Cleared cache: {stem_dir}")
