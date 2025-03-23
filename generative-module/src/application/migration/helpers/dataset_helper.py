import os
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import pretty_midi as pm
from transformers import PreTrainedTokenizerFast
import numpy as np
import torch

from application.encoder.services.encoder_service import encode_midi
from application.feature_extraction.services.feature_extraction_service import extract_symbolic_features
from application.encoder.models.vocab_model import RemiVocab, MoodsVocab, EmotionVocab
from domain.models.encoder.encoder_model import EncodeParameters
from domain.models.feature_extraction.feature_extraction_model import SymbolicFeaturesParameters
from persistence.dataloader.repositories.dataloader_repository import async_load
from application.migration.models.migration_model import DatasetParameters


def get_processed_files(processed_dir: Path) -> List[str]:
    """Get a list of processed MIDI files with their encoded data"""
    processed_files = []
    
    for file in os.listdir(processed_dir):
        if file.endswith(".processed.json"):
            midi_file = file.replace(".processed.json", "")
            processed_files.append(midi_file)
    
    return processed_files


def get_file_features(processed_dir: Path, midi_file: str) -> Dict:
    """Load the processed data for a MIDI file"""
    try:
        processed_data = async_load(processed_dir, midi_file, "processed")
        return processed_data
    except Exception as e:
        print(f"Error loading processed data for {midi_file}: {str(e)}")
        return None


def format_symbolic_features(features: Dict) -> str:
    """Format symbolic features as a text string"""
    # This is a simplified version - modify based on your specific feature format
    formatted = []
    
    # Add key features
    if "keys" in features and features["keys"].get("remi_keys"):
        key = features["keys"]["remi_keys"][0].pitch
        formatted.append(f"KEY={key}")
    
    # Add chords if available
    if "chords" in features and features["chords"].get("chords"):
        chords = [chord[2] for chord in features["chords"]["chords"][:5]]  # First 5 chords
        formatted.append(f"CHORDS={','.join(chords)}")
    
    # Add symbolic features if available
    if "symbolic_features" in features and features["symbolic_features"].get("piece_symbolic"):
        piece_features = features["symbolic_features"]["piece_symbolic"]
        # Add tempo
        if "avg_tempo" in piece_features:
            formatted.append(f"TEMPO={piece_features['avg_tempo']:.0f}")
        # Add note density
        if "note_density" in piece_features:
            formatted.append(f"DENSITY={piece_features['note_density']:.2f}")
    
    return " ".join(formatted)


def format_moods(moods: List[str]) -> str:
    """Format mood labels as text"""
    return f"MOOD={','.join(moods)}"


def remi_to_text(remi_events: List[str]) -> str:
    """Convert REMI events to a single space-separated string"""
    return " ".join(remi_events)


def create_text_dataset_entry(
    midi_file: str, 
    features: Dict, 
    moods: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Create a text dataset entry from a processed MIDI file
    Returns a dict with 'input' and 'output' keys
    """
    # Format the input prompt with features and moods
    input_parts = []
    
    # Add features
    if features:
        feature_text = format_symbolic_features(features)
        if feature_text:
            input_parts.append(feature_text)
    
    # Add moods if provided
    if moods:
        mood_text = format_moods(moods)
        input_parts.append(mood_text)
    
    # Format the input
    input_text = " ".join(input_parts)
    
    # Format the output (REMI events)
    if features and "encodings" in features and "events" in features["encodings"]:
        output_text = remi_to_text(features["encodings"]["events"])
    else:
        return None  # Skip if no encoding available
    
    return {
        "input": input_text,
        "output": output_text,
        "file": midi_file
    }


def create_dataset_from_processed(params: DatasetParameters) -> Dict[str, pd.DataFrame]:
    """
    Create text datasets from processed MIDI files
    Returns dict with 'train', 'validation', and 'test' dataframes
    """
    # Ensure output directory exists
    os.makedirs(params.output_dir, exist_ok=True)
    
    # Get list of processed files
    processed_files = get_processed_files(params.processed_dir)
    
    # Limit number of samples if specified
    if params.max_samples and len(processed_files) > params.max_samples:
        processed_files = random.sample(processed_files, params.max_samples)
    
    # Create dataset entries
    dataset_entries = []
    
    for midi_file in processed_files:
        # Load features
        features = get_file_features(params.processed_dir, midi_file)
        if not features:
            continue
        
        # Get moods if available
        moods = None
        if params.add_moods:
            # This would depend on how mood data is stored in your system
            # For now, generate some random moods for demonstration
            moods_vocab = MoodsVocab()
            mood_tokens = [moods_vocab.to_s(i) for i in range(len(moods_vocab))]
            moods = random.sample(mood_tokens, random.randint(1, 3))
        
        # Create dataset entry
        entry = create_text_dataset_entry(midi_file, features, moods)
        if entry:
            dataset_entries.append(entry)
    
    # Split dataset according to ratios
    random.shuffle(dataset_entries)
    
    train_size = int(len(dataset_entries) * params.split_ratio[0])
    val_size = int(len(dataset_entries) * params.split_ratio[1])
    
    train_entries = dataset_entries[:train_size]
    val_entries = dataset_entries[train_size:train_size + val_size]
    test_entries = dataset_entries[train_size + val_size:]
    
    # Create dataframes
    train_df = pd.DataFrame(train_entries)
    val_df = pd.DataFrame(val_entries)
    test_df = pd.DataFrame(test_entries)
    
    # Save to CSV
    train_df.to_csv(os.path.join(params.output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(params.output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(params.output_dir, "test.csv"), index=False)
    
    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df
    }


def tokenize_dataset(
    dataframes: Dict[str, pd.DataFrame],
    tokenizer: PreTrainedTokenizerFast,
    output_dir: Path,
    max_length: int = 1024
) -> Dict[str, Dict]:
    """
    Tokenize datasets and save them in a format suitable for training
    """
    tokenized_datasets = {}
    
    for split_name, df in dataframes.items():
        # Tokenize inputs
        inputs = tokenizer(
            df["input"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Tokenize outputs
        outputs = tokenizer(
            df["output"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Create dataset dict
        dataset_dict = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": outputs.input_ids,
            "decoder_attention_mask": outputs.attention_mask,
        }
        
        # Save tokenized dataset
        torch_path = os.path.join(output_dir, f"{split_name}_tokenized.pt")
        torch.save(dataset_dict, torch_path)
        
        tokenized_datasets[split_name] = dataset_dict
    
    return tokenized_datasets


def extract_features_from_processed(processed_data: Dict) -> Dict:
    """Extract symbolic and latent features from processed data"""
    features = {}
    
    # Extract symbolic features
    if "symbolic_features" in processed_data:
        if "bar_symbolic" in processed_data["symbolic_features"]:
            features["bar_symbolic"] = processed_data["symbolic_features"]["bar_symbolic"]
        if "piece_symbolic" in processed_data["symbolic_features"]:
            features["piece_symbolic"] = processed_data["symbolic_features"]["piece_symbolic"]
    
    # Extract latent features if available
    if "latents" in processed_data:
        features["latents"] = processed_data["latents"]
    
    return features


def create_projection_dataset_entry(
    midi_file: str,
    processed_data: Dict,
    moods: Optional[List[str]] = None,
    features_text: Optional[str] = None
) -> Dict:
    """
    Create a projection dataset entry from processed data
    This pairs text descriptions with symbolic/latent features
    """
    if not processed_data:
        return None
        
    # Extract features from processed data
    features = extract_features_from_processed(processed_data)
    if not features:
        return None
    
    # Format the text input (similar to create_text_dataset_entry)
    input_parts = []
    
    # Add features text if provided
    if features_text:
        input_parts.append(features_text)
    else:
        # Generate features text from processed data
        feature_text = format_symbolic_features(processed_data)
        if feature_text:
            input_parts.append(feature_text)
    
    # Add moods if provided
    if moods:
        mood_text = format_moods(moods)
        input_parts.append(mood_text)
    
    # Format the input
    input_text = " ".join(input_parts)
    
    return {
        "input_text": input_text,
        "midi_file": midi_file,
        "bar_symbolic": features.get("bar_symbolic"),
        "piece_symbolic": features.get("piece_symbolic"),
        "latents": features.get("latents"),
        "moods": moods
    }


def create_projection_dataset(params: DatasetParameters) -> Dict:
    """
    Create a dataset for training the projection model
    This creates text-to-features pairs for the projection stage
    """
    # Ensure output directory exists
    if params.features_output_dir:
        features_dir = params.features_output_dir
    else:
        features_dir = os.path.join(params.output_dir, "projection_data")
    
    os.makedirs(features_dir, exist_ok=True)
    
    # Get list of processed files
    processed_files = get_processed_files(params.processed_dir)
    
    # Limit number of samples if specified
    if params.max_samples and len(processed_files) > params.max_samples:
        processed_files = random.sample(processed_files, params.max_samples)
    
    # Create dataset entries
    projection_entries = []
    
    for midi_file in processed_files:
        # Load processed data
        processed_data = get_file_features(params.processed_dir, midi_file)
        if not processed_data:
            continue
        
        # Get moods if available
        moods = None
        if params.add_moods:
            # This would depend on how mood data is stored in your system
            # For now, generate some random moods for demonstration
            moods_vocab = MoodsVocab()
            mood_tokens = [moods_vocab.to_s(i) for i in range(len(moods_vocab))]
            moods = random.sample(mood_tokens, random.randint(1, 3))
        
        # Create projection dataset entry
        entry = create_projection_dataset_entry(midi_file, processed_data, moods)
        if entry:
            projection_entries.append(entry)
    
    # Split dataset according to ratios
    random.shuffle(projection_entries)
    
    train_size = int(len(projection_entries) * params.split_ratio[0])
    val_size = int(len(projection_entries) * params.split_ratio[1])
    
    train_entries = projection_entries[:train_size]
    val_entries = projection_entries[train_size:train_size + val_size]
    test_entries = projection_entries[train_size + val_size:]
    
    # Process and save the projection datasets
    projection_datasets = {
        "train": process_projection_entries(train_entries, os.path.join(features_dir, "train")),
        "validation": process_projection_entries(val_entries, os.path.join(features_dir, "validation")),
        "test": process_projection_entries(test_entries, os.path.join(features_dir, "test"))
    }
    
    # Create a metadata file with dataset info
    metadata = {
        "num_samples": {
            "train": len(train_entries),
            "validation": len(val_entries),
            "test": len(test_entries)
        },
        "feature_types": ["bar_symbolic", "piece_symbolic", "latents", "moods"]
    }
    
    with open(os.path.join(features_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    return projection_datasets


def process_projection_entries(entries: List[Dict], output_dir: str) -> Dict:
    """Process and save projection dataset entries"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract components
    input_texts = [entry["input_text"] for entry in entries]
    midi_files = [entry["midi_file"] for entry in entries]
    
    # Create text file with inputs and file mappings
    with open(os.path.join(output_dir, "inputs.txt"), "w") as f:
        for i, (text, midi) in enumerate(zip(input_texts, midi_files)):
            f.write(f"{i}\t{midi}\t{text}\n")
    
    # Process and save feature data
    features_data = {}
    
    # Process bar symbolic features
    bar_symbolic = [entry.get("bar_symbolic") for entry in entries]
    if any(bar_symbolic):
        # Filter None values and convert to appropriate format
        bar_symbolic_filtered = [bs for bs in bar_symbolic if bs is not None]
        if bar_symbolic_filtered:
            # Convert to tensor format or appropriate representation
            features_data["bar_symbolic"] = bar_symbolic_filtered
            torch.save(features_data["bar_symbolic"], os.path.join(output_dir, "bar_symbolic.pt"))
    
    # Process piece symbolic features
    piece_symbolic = [entry.get("piece_symbolic") for entry in entries]
    if any(piece_symbolic):
        # Filter None values and convert to appropriate format
        piece_symbolic_filtered = [ps for ps in piece_symbolic if ps is not None]
        if piece_symbolic_filtered:
            features_data["piece_symbolic"] = piece_symbolic_filtered
            torch.save(features_data["piece_symbolic"], os.path.join(output_dir, "piece_symbolic.pt"))
    
    # Process latent features
    latents = [entry.get("latents") for entry in entries]
    if any(latents):
        # Filter None values and convert to appropriate format
        latents_filtered = [l for l in latents if l is not None]
        if latents_filtered:
            features_data["latents"] = latents_filtered
            torch.save(features_data["latents"], os.path.join(output_dir, "latents.pt"))
    
    # Process mood features
    moods = [entry.get("moods") for entry in entries]
    if any(moods):
        # Filter None values and convert to appropriate format
        moods_filtered = [m for m in moods if m is not None]
        if moods_filtered:
            features_data["moods"] = moods_filtered
            with open(os.path.join(output_dir, "moods.json"), "w") as f:
                json.dump(moods_filtered, f)
    
    return features_data


def prepare_dataset(params: DatasetParameters, tokenizer: Optional[PreTrainedTokenizerFast] = None) -> Dict:
    """
    Prepare the dataset for training:
    1. Create text dataset from processed files
    2. Tokenize the dataset if a tokenizer is provided
    3. Create projection dataset if specified
    """
    result = {}
    
    # Create text dataset for generation
    if params.create_generation_data:
        dataframes = create_dataset_from_processed(params)
        result["text"] = dataframes
        
        # Tokenize if requested
        if params.tokenize and tokenizer:
            tokenized_data = tokenize_dataset(
                dataframes, 
                tokenizer, 
                params.output_dir,
                params.max_length
            )
            result["tokenized"] = tokenized_data
    
    # Create projection dataset if requested
    if params.create_projection_data:
        projection_data = create_projection_dataset(params)
        result["projection"] = projection_data
    
    return result 