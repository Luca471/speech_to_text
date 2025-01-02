"""Model loading utilities for the Speech-to-Text application."""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import streamlit as st

from config import WHISPER_MODEL

@st.cache_resource
def load_asr_model():
    """Load the Whisper ASR model and processor.
    
    Returns:
        tuple: (processor, model) or (None, None) if loading fails
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Always download from Hugging Face - don't try local first
        processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
        model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(device)
        
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None
