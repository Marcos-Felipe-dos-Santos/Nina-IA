"""Módulo de integração com o VTube Studio para avatar visual."""
from .emotion import detect_emotion
from .vtube import VTubeController, get_global_vtube

__all__ = ["detect_emotion", "VTubeController", "get_global_vtube"]
