"""Model loading utilities for the Speech-to-Text application."""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import streamlit as st
import warnings

from config import WHISPER_MODEL

# Filter out specific warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', message='.*Examining the path of torch.classes.*')

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
        model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL,
            # Add pad_token_id to avoid attention mask warning
            pad_token_id=processor.tokenizer.pad_token_id,
            # Use English as default language
            forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe")
        ).to(device)
        
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None
