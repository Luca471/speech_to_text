"""Configuration settings for the Speech-to-Text application."""

# Model settings
WHISPER_MODEL = "openai/whisper-base"
CHUNK_LENGTH_MS = 20000  # Length of audio chunks to process

# Audio settings
SAMPLE_RATE = 16000
NORMALIZE_HEADROOM = 0.1

# UI settings
TEXT_AREA_HEIGHT = 300
