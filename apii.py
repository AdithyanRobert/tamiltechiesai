import requests
import streamlit as st
import sounddevice as sd
import librosa
import numpy as np
import wave
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
from deep_translator import MyMemoryTranslator

# Ensure the required dependencies are installed
try:
    import torch
except ModuleNotFoundError:
    st.error("⚠️ The 'torch' module is missing. Please install it using 'pip install torch'.")
    st.stop()

# 🔑 Add your Mistral API Key here (Replace with your actual API key)
MISTRAL_API_KEY = "kikzWsaFDjKcHiV6gUCkp3jyUx6ojNqt"

# Initialize Whisper model (Force CPU mode if GPU is not available)
try:
    model = WhisperModel("medium", device="cpu", compute_type="float32")
except Exception as e:
    st.error(f"⚠️ Whisper Model Initialization Error: {e}")
    st.stop()

# Function to capture real-time audio from mic
def record_audio(filename="input.wav", duration=5, samplerate=16000):
    st.info("🎤 Listening... Speak Now!")
    
    try:
        # Capture audio
        recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()  # Wait until recording is finished
        
        # Save audio to a .wav file
        wav.write(filename, samplerate, recording)
        return filename
    except Exception as e:
        st.error(f"⚠️ Audio Recording Error: {e}")
        return None

# Function to query Mistral API
def query_mistral(prompt):
    url = "https://api.mistral.ai/v1/chat/completions"  # Correct Mistral API URL
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "open-mistral-7b",  # Use the correct model name
        "messages": [{"role": "user", "content": prompt}],  # Correct message format
        "temperature": 0.7
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No response received.")  # Extract response safely
        else:
            return f"⚠️ Mistral API Error: {response.text}"

    except requests.exceptions.RequestException as e:
        return f"⚠️ Error connecting to Mistral: {e}"

# Function to translate Tamil to English using MyMemory Translator
def translate_tamil_to_english(text):
    try:
        return MyMemoryTranslator(source="ta-IN", target="en-GB").translate(text)
    except Exception as e:
        st.error(f"⚠️ Translation Error (Tamil to English): {e}")
        return ""

# Function to translate English to Tamil using MyMemory Translator
def translate_english_to_tamil(text):
    try:
        return MyMemoryTranslator(source="en-GB", target="ta-IN").translate(text)
    except Exception as e:
        st.error(f"⚠️ Translation Error (English to Tamil): {e}")
        return ""

# Streamlit UI
st.title("🗣️ Tamil Speech AI Assistant")

if st.button("🎙️ Start Recording (5 sec)"):
    audio_path = record_audio()
    if audio_path:
        st.success("✅ Recording complete! Processing...")

        # Step 1: Convert Tamil Speech to Tamil Text
        try:
            segments, _ = model.transcribe(audio_path, language="ta")
            tamil_text = " ".join([segment.text for segment in segments])
            st.write("📝 **Transcribed Tamil Text:**", tamil_text)
        except Exception as e:
            st.error(f"⚠️ Error in Speech Recognition: {e}")
            tamil_text = ""
        
        # Step 2: Translate Tamil to English
        if tamil_text:
            english_query = translate_tamil_to_english(tamil_text)
            st.write("🔄 **Translated English Query:**", english_query)

            # Step 3: Get Mistral's response in English
            if english_query:
                english_response = query_mistral(english_query)
                st.write("🤖 **Mistral's English Response:**", english_response)

                # Step 4: Translate English Response back to Tamil
                tamil_response = translate_english_to_tamil(english_response)
                st.success("✅ Final Tamil Answer:")
                st.write("📝", tamil_response)
