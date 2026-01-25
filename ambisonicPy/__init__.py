"""
ambisonicPy - A Python Framework for ambisonic/spatial audio development
"""

from .speaker import Speaker
from .audio_processing import DistanceFilter
from .rendering import render_ambisonic_and_binaural
from .effects import (
    EFFECT_HANDLERS,
    apply_move,
)

__version__ = "0.1.0"

__all__ = [
    "Speaker",
    "DistanceFilter",
    "render_ambisonic_and_binaural",
    "EFFECT_HANDLERS",
    "apply_move",
]
  