"""Audio processing utilities for the Speech-to-Text application."""

from pydub import AudioSegment
import numpy as np
import streamlit as st

from config import SAMPLE_RATE, NORMALIZE_HEADROOM

def process_audio(file):
    """Process uploaded audio file for transcription.
    
    Args:
        file: Uploaded audio file
        
    Returns:
        AudioSegment or None if processing fails
    """
    try:
        # Read the uploaded file's content
        audio = AudioSegment.from_file(file)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Convert to target sample rate
        if audio.frame_rate != SAMPLE_RATE:
            audio = audio.set_frame_rate(SAMPLE_RATE)
            
        # Normalize audio
        normalized_audio = audio.normalize(headroom=NORMALIZE_HEADROOM)
        return normalized_audio
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def prepare_chunk(chunk):
    """Prepare audio chunk for model processing.
    
    Args:
        chunk: Audio chunk from AudioSegment
        
    Returns:
        numpy.ndarray: Processed audio samples
    """
    try:
        # Convert chunk to numpy array
        samples = np.array(chunk.get_array_of_samples())
        float_samples = samples.astype(np.float32) / 32768.0
        return float_samples
    except Exception as e:
        st.error(f"Error preparing audio chunk: {str(e)}")
        return None
