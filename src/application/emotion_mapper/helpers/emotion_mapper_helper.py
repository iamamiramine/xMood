import glob
import os
import torch

import pandas as pd

from src.domain.constants.encoder.token_constants import ANGER_KEY, LOVE_KEY, SADNESS_KEY, JOY_KEY, SURPRISE_KEY, BAR_KEY
from src.application.emotion_mapper.models.emotion_mapper_model import EmotionMapper
from src.domain.constants.paths_constants import LABELS_PATH, MIDI_PATH


def load_emotion_mapper_from_checkpoint(checkpoint_dir: str, eval=True):
    model, _ = EmotionMapper.load_checkpoint(checkpoint_dir, 1024, 512, eval)
    return model


def read_labels(dataset_name, split: str = None):
    path = os.path.join(LABELS_PATH, dataset_name, split, f"{split}_labels.csv") if split else os.path.join(LABELS_PATH, dataset_name, "labels.csv")
    labels = pd.read_csv(str(path))
    label_columns = labels.columns
    label_columns = label_columns.drop("file")
    midi_files = glob.glob(os.path.join(os.path.join(MIDI_PATH, dataset_name), "**/*.mid"), recursive=True)

    all_labels = []

    for i in midi_files:
        row = labels.loc[labels["file"] == (os.path.basename(i))]
        if not len(row):
            continue

        row_labels = {"file_name": str(os.path.basename(i))}
        for label_column in label_columns:
            row_labels[label_column] = row[label_column].values[0]
        all_labels.append(row_labels)

    all_labels = pd.DataFrame(all_labels)  # labels
    return all_labels, label_columns


def read_label_for_midi(dataset_name, midi_file_path, split: str = None):
    # Construct the path to the CSV label file
    path = os.path.join(LABELS_PATH, dataset_name, split, f"{split}_labels.csv") if split else os.path.join(LABELS_PATH, dataset_name, "labels.csv")
    labels = pd.read_csv(str(path))

    # Drop "file" from columns to isolate label columns
    label_columns = labels.columns.drop("file")

    # Extract only the basename of the MIDI file to match the CSV file format
    midi_file_name = os.path.basename(midi_file_path)

    # Locate the row corresponding to the specific MIDI file
    row = labels.loc[labels["file"] == midi_file_name]
    if not len(row):
        print(f"No labels found for {midi_file_name}")
        return None  # Return None if no label is found for this MIDI file

    # Prepare a dictionary for the MIDI file's labels
    midi_labels = {"file_name": midi_file_name}
    for label_column in label_columns:
        midi_labels[label_column] = row[label_column].values[0]

    # Convert to DataFrame for consistency with the original function's return format
    midi_labels_df = pd.DataFrame([midi_labels])
    return midi_labels_df, label_columns


def convert_emotions_to_sequence(emotions_vector, bar=None):
    """Convert emotion vector to a sequence of emotion tokens for a specific bar.

    Args:
        emotions_vector: List of 5 values representing [anger, joy, love, sadness, surprise]
        bar: Optional bar number to include in the sequence

    Returns:
        List of tokens in the format [BAR_X ANGER_Y LOVE_Z SADNESS_W JOY_V SURPRISE_U]
    """
    # Convert floating point values to integer percentages
    emotion_values = [int(e * 100) for e in emotions_vector]

    sequence = []
    if bar is not None:
        sequence.append(f"{BAR_KEY}_{bar}")

    # Add emotion tokens with their values
    emotion_keys = [ANGER_KEY, LOVE_KEY, SADNESS_KEY, JOY_KEY, SURPRISE_KEY]
    for key, value in zip(emotion_keys, emotion_values):
        sequence.append(f"{key}_{value}")

    return sequence
