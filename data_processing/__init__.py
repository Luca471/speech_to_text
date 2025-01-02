"""Data processing modules for the Speech-to-Text application."""

from .transcriber import transcribe_audio_in_chunks, process_chunk

__all__ = ['transcribe_audio_in_chunks', 'process_chunk']
