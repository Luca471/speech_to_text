"""Main Streamlit application for Speech-to-Text."""

import streamlit as st

from utils import load_asr_model, process_audio
from data_processing import transcribe_audio_in_chunks
from config import TEXT_AREA_HEIGHT

# Streamlit App Layout
st.title("Speech-to-Text")

# Initialize session state
if 'transcription' not in st.session_state:
    st.session_state.transcription = None

uploaded_file = st.file_uploader(
    "Upload your audio file", 
    type=["wav", "mp3", "mp4", "flac"]
)

if uploaded_file is not None and st.session_state.transcription is None:
    processor_model = load_asr_model()
    
    if processor_model[0] and processor_model[1]:
        with st.spinner("Processing audio..."):
            audio = process_audio(uploaded_file)
        
        if audio:
            transcription = transcribe_audio_in_chunks(audio, processor_model)
            if transcription:
                st.session_state.transcription = transcription

if st.session_state.transcription:
    # Show final transcription
    st.text_area("Transcription", 
                value=st.session_state.transcription,
                height=TEXT_AREA_HEIGHT,
                key="final_transcription")
    
    # Add download button for transcription
    st.download_button(
        label="Download transcription",
        data=st.session_state.transcription,
        file_name="transcription.txt",
        mime="text/plain"
    )
