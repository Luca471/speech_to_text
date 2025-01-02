# Speech-to-Text Transcription App

A Streamlit application that converts speech from audio files to text using OpenAI's Whisper model.

## Features

- Audio file upload support (WAV, MP3, MP4, FLAC)
- Progress tracking during transcription
- Downloadable transcription results
- Support for various audio formats and lengths

## Installation

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Install system dependencies:
   ```bash
   # On macOS
   brew install ffmpeg
   brew install libsndfile
   ```

## Project Structure

```
speech_to_text/
├── config.py                 # Configuration settings
├── streamlit_app.py         # Main Streamlit application
├── requirements.txt         # Python dependencies
├── packages.txt            # System dependencies
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── model_loader.py    # Model loading utilities
│   └── audio_processor.py # Audio processing utilities
└── data_processing/       # Data processing modules
    ├── __init__.py
    └── transcriber.py     # Transcription logic
```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Upload an audio file
3. Wait for the transcription to complete
4. Download or copy the transcribed text

## Dependencies

- Python packages: see requirements.txt
- System packages: see packages.txt
