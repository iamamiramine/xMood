"""Utility functions for integrating with feature_extraction and generator modules."""

import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

def convert_structured_features(feature_dict):
    """
    Convert structured features from the feature_extraction module 
    to a format compatible with the multimodal mapping service.
    
    Args:
        feature_dict: Dictionary containing structured features from feature_extraction
        
    Returns:
        features_array: Array of normalized features
    """
    # Define the feature keys that we'll use
    # This should match the features expected by the model
    feature_keys = [
        'mean_velocity', 
        'velocity_range',
        'note_density', 
        'rhythmic_complexity',
        'mean_note_duration',
        'pitch_range',
        'harmonic_complexity',
        'dissonance',
        'modality',      # E.g., major vs. minor 
        'tempo'
    ]
    
    # Initialize features array with zeros
    features = np.zeros(len(feature_keys))
    
    # Extract features and normalize them
    for i, key in enumerate(feature_keys):
        if key in feature_dict:
            # Get the value
            value = feature_dict[key]
            
            # Apply normalization based on expected ranges
            if key == 'mean_velocity':
                # Typically 0-127, normalize to 0-1
                features[i] = value / 127.0
            elif key == 'velocity_range':
                # Typically 0-127, normalize to 0-1
                features[i] = value / 127.0
            elif key == 'note_density':
                # Normalize based on reasonable max (e.g., 20 notes per beat)
                features[i] = min(value / 20.0, 1.0)
            elif key == 'rhythmic_complexity':
                # Assumed to be already normalized to 0-1
                features[i] = value
            elif key == 'mean_note_duration':
                # Normalize based on typical duration range (e.g., 0-4 beats)
                features[i] = min(value / 4.0, 1.0)
            elif key == 'pitch_range':
                # Normalize based on full MIDI range (0-127)
                features[i] = value / 127.0
            elif key == 'harmonic_complexity':
                # Assumed to be already normalized to 0-1
                features[i] = value
            elif key == 'dissonance':
                # Assumed to be already normalized to 0-1
                features[i] = value
            elif key == 'modality':
                # Often -1 (minor) to 1 (major), normalize to 0-1
                features[i] = (value + 1) / 2.0
            elif key == 'tempo':
                # Normalize based on typical range (e.g., 40-200 BPM)
                features[i] = (value - 40) / 160.0
                features[i] = max(0.0, min(features[i], 1.0))
    
    return features

def map_emotion_to_category(emotion_value):
    """
    Map emotional value to category index (0-4).
    Based on EmoMusicTV's emotion categorization.
    
    Args:
        emotion_value: Emotional value (typically valence, -1 to 1)
        
    Returns:
        category: Emotion category index (0-4)
    """
    # EmoMusicTV uses 5 categories:
    # Very negative (-2), moderate negative (-1), neutral (0), 
    # moderate positive (1), very positive (2)
    
    # Map to 0-4 range
    if emotion_value <= -0.6:
        return 0  # Very negative
    elif -0.6 < emotion_value <= -0.2:
        return 1  # Moderate negative
    elif -0.2 < emotion_value < 0.2:
        return 2  # Neutral
    elif 0.2 <= emotion_value < 0.6:
        return 3  # Moderate positive
    else:
        return 4  # Very positive

def format_features_for_generator(symbolic_features, vqvae_features):
    """
    Format the generated features to be compatible with the generator module.
    
    Args:
        symbolic_features: Numpy array of symbolic features [seq_len, batch_size, dim]
        vqvae_features: Numpy array of VQVAE features [seq_len, batch_size, dim]
        
    Returns:
        generator_input: Dictionary containing formatted features
    """
    # Combine features or format them as needed by the generator
    # The exact format will depend on what the generator expects
    
    # Example: Reshape and convert to format expected by generator
    # This is placeholder logic that needs to be adapted to match
    # the specific requirements of the generator module
    
    # Remove batch dimension (assuming batch_size=1)
    symbolic_features = symbolic_features[:, 0, :]
    vqvae_features = vqvae_features[:, 0, :]
    
    # Create the generator input format
    generator_input = {
        'symbolic_features': symbolic_features.tolist(),
        'vqvae_features': vqvae_features.tolist(),
        'num_bars': symbolic_features.shape[0],
        'feature_dim': symbolic_features.shape[1]
    }
    
    return generator_input

def save_features_to_file(features, filepath):
    """
    Save generated features to a JSON file.
    
    Args:
        features: Dictionary of features
        filepath: Path to save the file
    """
    # Convert numpy arrays to lists
    serializable_features = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            serializable_features[key] = value.tolist()
        else:
            serializable_features[key] = value
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(serializable_features, f, indent=2)
    
    logger.info(f"Features saved to {filepath}")

def load_features_from_file(filepath):
    """
    Load features from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        features: Dictionary of features
    """
    # Load from JSON file
    with open(filepath, 'r') as f:
        features = json.load(f)
    
    # Convert lists back to numpy arrays
    for key in ['symbolic_features', 'vqvae_features']:
        if key in features:
            features[key] = np.array(features[key])
    
    return features 