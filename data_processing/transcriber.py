"""Transcription logic for the Speech-to-Text application."""

import streamlit as st
import torch
from utils.audio_processor import prepare_chunk
from config import CHUNK_LENGTH_MS

def process_chunk(chunk, processor, model):
    """Process a single audio chunk for transcription.
    
    Args:
        chunk: Audio chunk from AudioSegment
        processor: Whisper processor
        model: Whisper model
        
    Returns:
        str: Transcribed text or empty string if failed
    """
    try:
        # Prepare audio samples
        float_samples = prepare_chunk(chunk)
        if float_samples is None:
            return ""
            
        # Process with Whisper
        inputs = processor(
            float_samples, 
            sampling_rate=16000, 
            return_tensors="pt",
            return_attention_mask=True  # Explicitly request attention mask
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate tokens with forced English
        with torch.no_grad():
            predicted_ids = model.generate(
                **inputs,
                max_length=256,
                no_repeat_ngram_size=3,
                num_beams=5
            )
        
        # Decode the tokens
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0] if transcription else ""
        
    except Exception as e:
        st.error(f"Error processing chunk: {str(e)}")
        return ""

def transcribe_audio_in_chunks(audio, processor_model_tuple, chunk_length_ms=CHUNK_LENGTH_MS):
    """Transcribe audio file in chunks with progress tracking.
    
    Args:
        audio: Processed AudioSegment
        processor_model_tuple: Tuple of (processor, model)
        chunk_length_ms: Length of each chunk in milliseconds
        
    Returns:
        str: Complete transcription or None if failed
    """
    try:
        processor, model = processor_model_tuple
        if not processor or not model:
            st.error("Model or processor not loaded correctly")
            return None
        
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("0% completed")
        
        # Process chunks
        total_chunks = (len(audio) + chunk_length_ms - 1) // chunk_length_ms
        processed_chunks = 0
        ordered_transcriptions = []
        
        for start_ms in range(0, len(audio), chunk_length_ms):
            chunk = audio[start_ms:start_ms + chunk_length_ms]
            transcription = process_chunk(chunk, processor, model)
            
            if transcription and transcription.strip():
                ordered_transcriptions.append(transcription)
                processed_chunks += 1
                
                # Update progress
                progress = processed_chunks / total_chunks
                progress_bar.progress(progress)
                status_text.text(f"{int(progress * 100)}% completed")
        
        # Final transcription
        full_transcription = " ".join(ordered_transcriptions)
        if full_transcription.strip():
            progress_bar.progress(1.0)
            status_text.text("100% completed")
            return full_transcription
            
        st.error("No transcription generated")
        return None
            
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None
