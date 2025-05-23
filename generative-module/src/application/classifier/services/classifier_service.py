import os
import traceback
from typing import Dict, Any, List, Union, Optional
import numpy as np
from pathlib import Path
import json
import pickle
import pandas as pd
import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
from argparse import Namespace
from tqdm import tqdm
import asyncio

# Import functionality from existing service files
from application.classifier.services.train_test import main as train_main
from application.classifier.helpers.preprocessing import midi_feature_extract
from application.classifier.models.model.net import SAN
from application.classifier.helpers.midi_helper.remi.midi2event import analyzer, corpus, event
from application.classifier.models.data import PEmo_Dataset

from domain.models.classifier.classifier_model import PredictMoodParameters, TrainingParameters, MidiFeatureExtractParameters, BatchPredictParameters

# Define mood names for the 9 mood categories
MOOD_NAMES = ["HAPPY_KEY", "DRAMATIC_KEY", "RELAXING_KEY", "LOVE_KEY", "DARK_KEY", "CHRISTMAS_KEY", "ENERGETIC_KEY", "MEDITATIVE_KEY", "MOTIVATIONAL_KEY"]

# Path configurations
path_data_root = "datasets/EMOPIA/"
path_dictionary = os.path.join(path_data_root, "dictionary.pkl")

# Load dictionary if exists
try:
    midi_dictionary = pickle.load(open(path_dictionary, "rb"))
    event_to_int = midi_dictionary[0]
except (FileNotFoundError, EOFError):
    event_to_int = {}


def predict(
    midi_path: str, model_type: str = "remi", task: str = "mood", device: str = "cpu", parameters: Optional[PredictMoodParameters] = None
) -> Dict[str, Any]:
    """
    Predict mood or emotion for a MIDI file.

    Parameters:
    -----------
    midi_path : str
        Path to the MIDI file
    model_type : str
        Model type (remi, midi_like, wav)
    task : str
        Prediction task ("mood" or "emotion")
    device : str
        Device to run inference on (cpu, cuda:0)
    parameters : PredictMoodParameters, optional
        Optional typed parameters object (alternative to individual parameters)

    Returns:
    --------
    dict
        A dictionary containing prediction results
    """
    try:
        # Use parameters object if provided
        if parameters:
            midi_path = parameters.midi_path
            model_type = parameters.model_type
            device = parameters.device
            task = "mood"  # Force mood task if using parameters

        # Helper function for REMI extraction
        def remi_extractor(midi_path, event_to_int):
            # Create a custom Note class that allows setting duration
            class NoteWithDuration:
                def __init__(self, note):
                    self.start = note.start
                    self.end = note.end
                    self.pitch = note.pitch
                    self.velocity = note.velocity
                    self.instr_idx = getattr(note, "instr_idx", 0)
                    self.duration = 0
                    self.shift = 0

            midi_obj = analyzer(midi_path)

            # Convert notes to custom note objects that allow setting duration
            for instr in midi_obj.instruments:
                new_notes = []
                for note in instr.notes:
                    new_notes.append(NoteWithDuration(note))
                instr.notes = new_notes

            song_data = corpus(midi_obj)
            event_sequence = event(song_data)

            quantize_midi = []
            for i in event_sequence:
                event_key = str(i["name"]) + "_" + str(i["value"])
                try:
                    quantize_midi.append(event_to_int[event_key])
                except KeyError:
                    # Special handling for velocity values
                    if "Velocity" in i["name"]:
                        # Try multiple fallback strategies for velocity
                        found = False

                        # First try with adjusted values
                        for offset in range(-10, 11, 2):  # Try offsets between -10 and +10 in steps of 2
                            adjusted_key = f"{i['name']}_{i['value'] + offset}"
                            if adjusted_key in event_to_int:
                                quantize_midi.append(event_to_int[adjusted_key])
                                found = True
                                break

                        # If still not found, find closest available velocity
                        if not found:
                            # Extract all velocity values available in the dictionary
                            velocity_keys = [k for k in event_to_int.keys() if "Velocity" in k]
                            if velocity_keys:
                                # Find the closest velocity value
                                velocity_values = [int(k.split("_")[-1]) for k in velocity_keys]
                                target_velocity = i["value"]
                                closest_velocity = min(velocity_values, key=lambda x: abs(x - target_velocity))
                                closest_key = f"{i['name']}_{closest_velocity}"
                                if closest_key in event_to_int:
                                    quantize_midi.append(event_to_int[closest_key])
                                    found = True

                            if not found:
                                # Skip if we still can't find a good match
                                print(f"Warning: Could not find a suitable replacement for {event_key}")
                                continue
                    else:
                        # For non-velocity events, just skip unknown events
                        print(f"Warning: Unknown event {event_key}, skipping")
                        continue

            return quantize_midi

        # Set device
        if device != "cpu" and not device.startswith("cuda"):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if device != "cpu" and torch.cuda.is_available():
            print("GPU name:", torch.cuda.get_device_name(device=device))

        # Load model configuration
        config_path = os.path.join("output", "demos", "demo_2", "classifier", "ReMIDICaps", task, model_type, "hparams.yaml")
        checkpoint_path = os.path.join("output", "demos", "demo_2", "classifier", "ReMIDICaps", task, model_type, "last-v1.ckpt")
        config = OmegaConf.load(config_path)

        # Determine label list based on task
        if task == "mood":
            label_list = MOOD_NAMES
            cls_type = "MOOD"
        else:
            label_list = list(config.task.labels)
            cls_type = config.task.cls_type if hasattr(config.task, "cls_type") else "AV"

        # Initialize model
        model = SAN(
            num_of_dim=config.task.num_of_dim,
            vocab_size=1382,  # Changed from config.midi.pad_idx + 1 to match checkpoint
            lstm_hidden_dim=128,  # Changed from config.hparams.lstm_hidden_dim to match checkpoint (512/4=128)
            embedding_size=config.hparams.embedding_size,
            r=config.hparams.r,
            cls_type=cls_type,
        )

        # Load model weights
        state_dict = torch.load(checkpoint_path, map_location=torch.device(device))
        new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
        new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
        model.load_state_dict(new_state_dict)
        model.eval()
        model = model.to(device)

        # Process input
        quantize_midi = remi_extractor(midi_path, event_to_int)
        model_input = torch.LongTensor(quantize_midi).unsqueeze(0)
        prediction = model(model_input.to(device))

        # Process prediction results
        if task == "mood":
            # Handle mood prediction output
            pred_values = prediction.squeeze(0).detach().cpu().numpy()
            pred_label_idx = np.argmax(pred_values)
            pred_label = label_list[pred_label_idx]

            # Create mood probabilities dictionary
            mood_probs = {mood: float(pred_values[i]) for i, mood in enumerate(label_list)}

            return {
                "status": "success",
                "file_name": os.path.basename(midi_path),
                "dominant_mood": pred_label,
                "dominant_probability": float(pred_values[pred_label_idx]),
                "mood_probabilities": mood_probs,
            }
        else:
            # Handle valence-arousal prediction
            pred_label = label_list[prediction.squeeze(0).max(0)[1].detach().cpu().numpy()]
            pred_values = prediction.squeeze(0).detach().cpu().numpy()

            return {"status": "success", "file_name": os.path.basename(midi_path), "emotion": pred_label, "values": pred_values.tolist()}

    except Exception as e:
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}


def train(
    parameters: Union[TrainingParameters, MidiFeatureExtractParameters, Dict[str, Any]] = None, extract_features: bool = False, **kwargs
) -> Dict[str, Any]:
    """
    Train a classifier model or extract MIDI features.

    Parameters:
    -----------
    parameters : Union[TrainingParameters, MidiFeatureExtractParameters, Dict], optional
        Training or feature extraction parameters
    extract_features : bool
        Whether to extract features (True) or train a model (False)
    **kwargs : Any
        Additional keyword arguments for training or feature extraction

    Returns:
    --------
    dict
        A dictionary with training or extraction status
    """
    try:
        # Extract features if requested
        if extract_features:
            # Convert parameters to correct type if needed
            if isinstance(parameters, dict):
                extract_params = MidiFeatureExtractParameters(**parameters)
            elif parameters is not None:
                extract_params = parameters

            # Extract MIDI features
            processed_files = midi_feature_extract(
                midi_path=extract_params.midi_path,
                remi_path=extract_params.remi_path,
                csv_path=extract_params.csv_path,
                dictionary_path=extract_params.dictionary_path,
            )

            return {"status": "success", "message": f"Successfully processed {processed_files} MIDI files"}
        else:
            # Convert parameters to correct type if needed
            if isinstance(parameters, dict):
                train_params = parameters
            elif parameters is not None:
                train_params = parameters.dict()

            # Train model
            train_main(**train_params)
            return {"status": "success", "message": "Model training completed successfully"}

    except Exception as e:
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}


async def predict_batch(parameters: BatchPredictParameters) -> Dict[str, Any]:
    """
    Batch predict mood for MIDI files listed in a CSV and compare with ground truth.

    Parameters:
    -----------
    parameters : BatchPredictParameters
        Parameters for batch prediction including:
        - csv_path: Path to CSV containing MIDI filenames and ground truth
        - midi_dir: Directory containing MIDI files
        - model_type: Model type (remi, midi_like, wav)
        - batch_size: Number of files to process in parallel
        - device: Device to run inference on
        - output_dir: Directory to save prediction results (optional)

    Returns:
    --------
    dict
        A dictionary containing prediction results and statistics
    """
    # Load CSV file
    print(f"Loading dataset from {parameters.csv_path}")
    try:
        df = pd.read_csv(parameters.csv_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load CSV: {str(e)}"}

    # Create output directory if specified
    if parameters.output_dir:
        os.makedirs(parameters.output_dir, exist_ok=True)
        results_path = os.path.join(parameters.output_dir, "prediction_results.csv")

    # Track statisticsq
    total_files = len(df)
    processed = 0
    successful = 0
    correct_top_predictions = 0
    all_predictions = []
    errors = []
    prediction_results = []

    # Get key from CSV file if present or use default key column
    mood_column = parameters.mood_column if hasattr(parameters, "mood_column") and parameters.mood_column else "mood_tokens"
    file_column = parameters.file_column if hasattr(parameters, "file_column") and parameters.file_column else "file"

    # Check if required columns exist
    if file_column not in df.columns:
        return {"status": "error", "message": f"Required column '{file_column}' not found in CSV"}

    has_ground_truth = mood_column in df.columns
    if not has_ground_truth:
        print(f"Warning: Ground truth column '{mood_column}' not found in CSV. Only predictions will be recorded.")

    async def process_file(row) -> Dict[str, Any]:
        try:
            # Get file path
            file_name = row[file_column]
            if not isinstance(file_name, str):
                return {"status": "error", "message": f"Invalid filename: {file_name}"}

            # If the file name is already a full path, use it directly, otherwise join with midi_dir
            if os.path.isabs(file_name) and os.path.exists(file_name):
                midi_file_path = file_name
            else:
                # Get just the filename if it's a full path
                file_name = os.path.basename(file_name)
                midi_file_path = os.path.join(parameters.midi_dir, file_name)

            # Check if file exists
            if not os.path.exists(midi_file_path):
                return {"status": "error", "file_name": file_name, "message": f"File not found: {midi_file_path}"}

            # Create prediction parameters
            predict_params = PredictMoodParameters(midi_path=midi_file_path, model_type=parameters.model_type, device=parameters.device)

            # Get prediction
            prediction_result = predict(midi_path=midi_file_path, model_type=parameters.model_type, device=parameters.device)

            # Add ground truth to result if available
            if has_ground_truth and mood_column in row:
                mood_tokens_str = row[mood_column]

                # Parse mood tokens string - format is: "Mood_X_0.123 Mood_Y_0.456 ..."
                if isinstance(mood_tokens_str, str):
                    try:
                        # Split by spaces to get individual mood tokens
                        mood_tokens = mood_tokens_str.split()

                        # Parse each token
                        ground_truth_moods = {}
                        for token in mood_tokens:
                            # Format: Mood_X_0.123
                            parts = token.split("_")
                            if len(parts) >= 3:
                                # Get the mood name and probability
                                mood_name = f"{parts[0]}_{parts[1]}"
                                try:
                                    prob_value = float(parts[2])
                                    ground_truth_moods[mood_name] = prob_value
                                except ValueError:
                                    # Skip if probability isn't a valid float
                                    continue

                        # Get the top mood (highest probability) from ground truth
                        if ground_truth_moods:
                            top_ground_truth_mood = max(ground_truth_moods.items(), key=lambda x: x[1])[0]

                            # Store parsed ground truth in result
                            prediction_result["ground_truth_moods"] = ground_truth_moods
                            prediction_result["top_ground_truth_mood"] = top_ground_truth_mood

                            # Check if top predicted mood matches top ground truth mood
                            prediction_result["correct_top_mood"] = prediction_result["dominant_mood"] == top_ground_truth_mood
                    except Exception as e:
                        prediction_result["ground_truth_parse_error"] = str(e)

            prediction_result["file_name"] = file_name
            return prediction_result

        except Exception as e:
            return {"status": "error", "file_name": row[file_column] if file_column in row else "unknown", "message": str(e)}

    # Process files in batches
    batch_size = parameters.batch_size
    for i in range(0, total_files, batch_size):
        batch_rows = df.iloc[i : i + batch_size].to_dict("records")
        batch_tasks = [process_file(row) for row in batch_rows]

        # Process batch using asyncio
        results = await asyncio.gather(*batch_tasks, return_exceptions=False)

        # Update statistics and collect results
        for result in results:
            if result["status"] == "success":
                successful += 1
                prediction_results.append(result)
                all_predictions.append(result["dominant_mood"])

                # Count correct predictions if ground truth is available and top mood matches
                if has_ground_truth and result.get("correct_top_mood", False):
                    correct_top_predictions += 1
            else:
                errors.append(result)
                prediction_results.append(result)

        processed += len(batch_rows)
        print(f"Processed {processed}/{total_files} files", flush=True)

    # Calculate accuracy if ground truth is available
    top_mood_accuracy = correct_top_predictions / successful if successful > 0 and has_ground_truth else None

    # Calculate mood distribution in predictions
    mood_distribution = {}
    for mood in all_predictions:
        if mood in mood_distribution:
            mood_distribution[mood] += 1
        else:
            mood_distribution[mood] = 1

    # Convert to percentages
    if all_predictions:
        mood_distribution = {mood: count / len(all_predictions) for mood, count in mood_distribution.items()}

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(prediction_results)
    results_df.to_csv(results_path, index=False)

    # Prepare summary
    summary = {
        "status": "success",
        "total_files": total_files,
        "successful": successful,
        "failed": len(errors),
        "results_path": results_path,
        "mood_distribution": mood_distribution,
    }

    if has_ground_truth:
        summary["correct_top_predictions"] = correct_top_predictions
        summary["top_mood_accuracy"] = top_mood_accuracy

    # Include detailed errors if needed
    if errors:
        error_summary = [f"{e.get('file_name', 'unknown')}: {e.get('message', 'Unknown error')}" for e in errors[:10]]
        if len(errors) > 10:
            error_summary.append(f"... and {len(errors) - 10} more errors")
        summary["errors"] = error_summary

    return summary


def predict_batch_sync(parameters: BatchPredictParameters) -> Dict[str, Any]:
    """
    Synchronous wrapper for predict_batch async function.

    Parameters:
    -----------
    parameters : BatchPredictParameters
        Parameters for batch prediction

    Returns:
    --------
    dict
        A dictionary containing prediction results and statistics
    """
    import asyncio

    # Create a new event loop if needed
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run the async function in the event loop
    return loop.run_until_complete(predict_batch(parameters))
