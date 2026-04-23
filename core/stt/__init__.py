# core.stt — Módulo de Speech-to-Text
from core.stt.transcriber import WhisperTranscriber
from core.stt.microphone import MicrophoneCapture

__all__ = ["WhisperTranscriber", "MicrophoneCapture"]
