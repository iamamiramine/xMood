import torch
import numpy as np
import re
from typing import List, Dict, Optional, Tuple, Union

# Common instrument list based on General MIDI program numbers
COMMON_INSTRUMENTS = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone", 
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ", 
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)",
    "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics",
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    "Violin", "Viola", "Cello", "Contrabass", 
    "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
    "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", 
    "Oboe", "English Horn", "Bassoon", "Clarinet",
    "Piccolo", "Flute", "Recorder", "Pan Flute", 
    "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
    "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)",
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
    "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    "Sitar", "Banjo", "Shamisen", "Koto", 
    "Kalimba", "Bagpipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", 
    "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet", 
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
    "drum"  # Special case for drums
]

# Common chord types
COMMON_CHORDS = [
    # Major and minor triads
    "maj", "min", "aug", "dim",
    # Sevenths
    "maj7", "min7", "dom7", "dim7", "hdim7", "minmaj7",
    # Added tone chords
    "add9", "add11", "add13",
    # Sixths
    "maj6", "min6",
    # Ninths
    "maj9", "min9", "dom9",
    # Elevenths
    "maj11", "min11", "dom11",
    # Thirteenths
    "maj13", "min13", "dom13",
    # Suspended chords
    "sus2", "sus4",
    # Special case
    "N"  # No chord
]

# Common keys (24 major and minor keys)
COMMON_KEYS = [
    "C:maj", "C:min", "C#:maj", "C#:min", "D:maj", "D:min", 
    "D#:maj", "D#:min", "E:maj", "E:min", "F:maj", "F:min",
    "F#:maj", "F#:min", "G:maj", "G:min", "G#:maj", "G#:min", 
    "A:maj", "A:min", "A#:maj", "A#:min", "B:maj", "B:min"
]

# Common time signatures
COMMON_TIME_SIGNATURES = ["2/4", "3/4", "4/4", "6/8", "9/8", "12/8"]

def parse_mood_tokens(mood_tokens_str: str) -> Tuple[List[str], List[float]]:
    """
    Parse mood tokens string from the ReMIDICaps dataset.
    
    Args:
        mood_tokens_str: String of mood tokens in format "Mood_Name_Weight Mood_Name_Weight ..."
                        E.g., "Mood_Relaxing_0.2844 Mood_Christmas_0.2736 Mood_Dramatic_0.2423"
    
    Returns:
        Tuple of (mood_names, mood_weights)
    """
    if not mood_tokens_str or isinstance(mood_tokens_str, float):  # Handle NaN values
        return [], []
        
    tokens = mood_tokens_str.strip().split()
    mood_names = []
    mood_weights = []
    
    for token in tokens:
        parts = token.split('_')
        if len(parts) >= 3:
            mood_name = '_'.join(parts[1:-1])  # Handle mood names with multiple underscores
            try:
                weight = float(parts[-1])
                mood_names.append(mood_name)
                mood_weights.append(weight)
            except ValueError:
                # Skip malformed tokens
                continue
    
    return mood_names, mood_weights

def process_global_features(global_features_str: str) -> Dict[str, Union[str, float]]:
    """
    Process global features string from the ReMIDICaps dataset.
    
    Args:
        global_features_str: String of global features in format "Feature_Value Feature_Value ..."
                            E.g., "Time Signature_4/4 Key Signature_D:maj Note Density_2 ..."
    
    Returns:
        Dictionary of feature names to values
    """
    if not global_features_str or isinstance(global_features_str, float):  # Handle NaN values
        return {}
        
    features = {}
    
    # Split by space, but handle feature names with spaces
    # This is a bit tricky because we need to handle features like "Time Signature_4/4"
    # where "Time Signature" is the feature name and "4/4" is the value
    
    # First, try to split by recognizing the pattern of Feature_Value
    feature_patterns = re.findall(r'([A-Za-z\s]+)_([A-Za-z0-9:/.#]+)', global_features_str)
    
    for name, value in feature_patterns:
        name = name.strip()
        
        # Handle numeric values
        if re.match(r'^-?\d+(\.\d+)?$', value):
            features[name] = float(value)
        else:
            features[name] = value
            
    return features

def create_mood_vector(mood_tokens_str: str, all_moods: List[str] = None) -> torch.Tensor:
    """
    Create a mood vector from mood tokens string.
    
    Args:
        mood_tokens_str: String of mood tokens in format "Mood_Name_Weight Mood_Name_Weight ..."
        all_moods: Optional list of all possible moods for consistent ordering
                  If not provided, will use the moods found in the string
    
    Returns:
        Tensor of mood weights
    """
    mood_names, mood_weights = parse_mood_tokens(mood_tokens_str)
    
    if not mood_names:
        # Return zero vector if no moods or empty string
        return torch.zeros(len(all_moods) if all_moods else 1)
    
    if all_moods is None:
        # If no list of all moods provided, just return the weights in order
        return torch.tensor(mood_weights, dtype=torch.float)
    
    # Create vector with consistent ordering based on all_moods
    mood_vector = torch.zeros(len(all_moods), dtype=torch.float)
    for name, weight in zip(mood_names, mood_weights):
        if name in all_moods:
            mood_vector[all_moods.index(name)] = weight
    
    return mood_vector

def create_global_feature_vector(global_features_str: str, feature_config: Dict = None) -> torch.Tensor:
    """
    Create a global feature vector from global features string.
    
    Args:
        global_features_str: String of global features
        feature_config: Configuration for feature processing with:
                       - 'numeric_features': List of feature names to treat as numeric
                       - 'categorical_features': Dict of feature names to possible values
                       - 'instruments': List of instrument names 
                       - 'chords': List of chord types
    
    Returns:
        Tensor of global features
    """
    features_dict = process_global_features(global_features_str)
    
    if not features_dict:
        # Return zero vector if no features or empty string
        return torch.zeros(64)  # Default size
    
    if feature_config is None:
        # Default configuration with predefined categories
        feature_config = {
            'numeric_features': ['Note Density', 'Mean Velocity', 'Mean Pitch', 'Mean Duration'],
            'categorical_features': {
                'Time Signature': COMMON_TIME_SIGNATURES,
                'Key Signature': COMMON_KEYS
            },
            'instruments': COMMON_INSTRUMENTS,
            'chords': COMMON_CHORDS
        }
    
    # Process numeric features
    numeric_values = []
    for feature in feature_config.get('numeric_features', []):
        if feature in features_dict:
            value = features_dict[feature]
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    value = 0.0
            numeric_values.append(value)
        else:
            numeric_values.append(0.0)
    
    # Process categorical features
    categorical_vectors = []
    for feature, possible_values in feature_config.get('categorical_features', {}).items():
        if feature in features_dict:
            value = features_dict[feature]
            one_hot = [1.0 if value == pv else 0.0 for pv in possible_values]
        else:
            one_hot = [0.0] * len(possible_values)
        categorical_vectors.extend(one_hot)
    
    # Process instruments as categorical data with predefined list
    instrument_vector = []
    if 'instruments' in feature_config:
        instruments_list = feature_config['instruments']
        # Extract instruments from features_dict
        found_instruments = [k.replace('Instrument_', '') for k in features_dict.keys() 
                          if k.startswith('Instrument_')]
        
        # Create one-hot encoding for each instrument
        instrument_vector = [1.0 if instr in found_instruments else 0.0 
                           for instr in instruments_list]
    
    # Process chords as categorical data with predefined list
    chord_vector = []
    if 'chords' in feature_config:
        chord_types = feature_config['chords']
        
        # Extract all chords from features_dict and separate into components
        chord_entries = [k.replace('Chord_', '') for k in features_dict.keys() 
                      if k.startswith('Chord_')]
        
        # Parse into root and type (e.g., "C:maj" → root="C", type="maj")
        found_chord_types = set()
        for chord in chord_entries:
            parts = chord.split(':')
            if len(parts) >= 2:
                chord_type = parts[1]
                found_chord_types.add(chord_type)
        
        # Create one-hot encoding for each chord type
        chord_vector = [1.0 if chord_type in found_chord_types else 0.0 
                       for chord_type in chord_types]
    
    # Combine all features
    all_features = numeric_values + categorical_vectors + instrument_vector + chord_vector
    
    # Convert to tensor
    return torch.tensor(all_features, dtype=torch.float)

def process_remidi_sample(
    caption: str = None, 
    mood_tokens: str = None, 
    global_features: str = None,
    all_moods: List[str] = None,
    feature_config: Dict = None
) -> Dict[str, torch.Tensor]:
    """
    Process a sample from the ReMIDICaps dataset.
    
    Args:
        caption: Text caption describing the music
        mood_tokens: String of mood tokens
        global_features: String of global features
        all_moods: Optional list of all possible moods for consistent ordering
        feature_config: Configuration for feature processing
    
    Returns:
        Dictionary with processed features:
        - 'text_prompts': The original caption
        - 'moods': Tensor of mood weights
        - 'global_features': Tensor of global features
    """
    result = {}
    
    if caption is not None and not isinstance(caption, float):
        result['text_prompts'] = caption
    
    if mood_tokens is not None:
        result['moods'] = create_mood_vector(mood_tokens, all_moods)
    
    if global_features is not None:
        result['global_features'] = create_global_feature_vector(global_features, feature_config)
    
    return result 