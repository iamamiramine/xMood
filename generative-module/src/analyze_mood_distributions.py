import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import pickle
from pathlib import Path


def extract_midi_statistics(processed_dir, csv_path):
    # Initialize counters and lists
    total_bars = []
    all_chords = set()
    all_keys = set()
    all_instruments = set()
    
    # Get all pickle files
    pickle_files = list(Path(processed_dir).rglob("*.pkl"))
    num_pieces = len(pickle_files)
    
    # Process each pickle file
    for pkl_file in pickle_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            
        bar_symbolic = data['symbolic_features']['bar_symbolic']
        
        # Count bars
        bar_count = sum(1 for token in bar_symbolic if token.startswith('Bar_'))
        total_bars.append(bar_count)
        
        # Extract chords
        chords = {token.replace('Chord_', '') for token in bar_symbolic if token.startswith('Chord_')}
        all_chords.update(chords)
        
        # Extract key signatures
        keys = {token.replace('Key Signature_', '') for token in bar_symbolic if token.startswith('Key Signature_')}
        all_keys.update(keys)
        
        # Extract instruments
        instruments = {token.replace('Instrument_', '') for token in bar_symbolic if token.startswith('Instrument_')}
        all_instruments.update(instruments)
    
    # Calculate average bars
    avg_bars = sum(total_bars) / len(total_bars) if total_bars else 0
    
    # Read CSV file for genres
    df = pd.read_csv(csv_path)
    # # Split genres by semicolon and get unique values
    # all_genres = set()
    # for genres in df['genre'].str.split(';'):
    #     if genres is not None:  # Handle potential NaN values
    #         all_genres.update(genres)
    
    # Compile statistics
    statistics = {
        "Number of MIDI Pieces": num_pieces,
        "Average Length (Bars)": round(avg_bars, 2),
        "Number of Chord Types": len(all_chords),
        "Number of Key Types": len(all_keys),
        "Number of Instruments": len(all_instruments),
        # "Number of Genres": len(all_genres)
    }
    
    return statistics

def main():
    # Define paths
    processed_dir = "../../output/demos/demo_2/processed/ReMIDICaps"
    csv_path = "../../datasets/ReMIDICaps/labels/ReMIDICaps.csv"
    
    # Extract statistics
    stats = extract_midi_statistics(processed_dir, csv_path)
    
    # Print results
    print("\nMIDI Dataset Statistics:")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()