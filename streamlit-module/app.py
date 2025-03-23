import streamlit as st
import requests
import os
from typing import Dict, Any

# Configure the page
st.set_page_config(page_title="Music Generation Pipeline", page_icon="🎵", layout="wide")

# Constants
API_URL = "http://pa-ai:80"  # This will be the service name in docker-compose


def generate_from_midi(params: Dict[str, Any]) -> Dict[str, Any]:
    """Send a request to the API to generate music from MIDI."""
    response = requests.post(f"{API_URL}/generator/generate_from_midi", json=params)
    return response.json()


def main():
    st.title("🎵 Music Generation Pipeline")
    st.markdown("Generate music using MIDI files as input")

    with st.form("generation_form"):
        col1, col2 = st.columns(2)

        with col1:
            latent_midi_path = st.text_input("Latent MIDI Path", help="Path to the latent MIDI file")
            symbolic_midi_path = st.text_input("Symbolic MIDI Path", help="Path to the symbolic MIDI file")
            emotions_midi_path = st.text_input("Emotions MIDI Path", help="Path to the emotions MIDI file")
            output_folder = st.text_input("Output Folder", help="Folder where the generated file will be saved")
            output_name = st.text_input("Output Name", help="Name for the generated file")

        with col2:
            checkpoint_path = st.text_input("Checkpoint Path", help="Path to the generator checkpoint")
            context_size = st.number_input("Context Size", min_value=1, value=256, help="Size of the context window")
            max_bars = st.number_input("Maximum Bars", min_value=1, value=16, help="Maximum number of bars to generate")
            max_positions = st.number_input("Maximum Positions", min_value=1, value=512, help="Maximum number of positions")
            max_n_tokens = st.number_input("Maximum Tokens", min_value=1, value=1024, help="Maximum number of tokens to generate")
            temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.1, help="Sampling temperature (higher = more random)")

        submitted = st.form_submit_button("Generate Music")

        if submitted:
            with st.spinner("Generating music..."):
                try:
                    params = {
                        "latent_midi": latent_midi_path,
                        "symbolic_midi": symbolic_midi_path,
                        "emotions_midi": emotions_midi_path,
                        "output_folder": output_folder,
                        "output_name": output_name,
                        "checkpoint_path": checkpoint_path,
                        "context_size": context_size,
                        "max_bars": max_bars,
                        "max_positions": max_positions,
                        "max_n_tokens": max_n_tokens,
                        "temperature": temperature,
                    }

                    result = generate_from_midi(params)
                    st.success("Music generated successfully!")
                    st.json(result)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
