"""
ambisonicPy - A Python Framework for ambisonic/spatial audio development
"""

from .speaker import Speaker
from .soundstage import SoundStage
from .audio_processing import DistanceFilter
from .effects import (
    EFFECT_HANDLERS,
    apply_move,
)

__version__ = "0.1.0"

__all__ = [
    "Speaker",
    "SoundStage",
    "DistanceFilter",
    "EFFECT_HANDLERS",
    "apply_move",
]
  