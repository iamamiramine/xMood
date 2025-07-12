import os
import json
import numpy as np
import pandas as pd
import torch

import random
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from domain.constants.model_constants import ModelConstants


class PEmo_Dataset(Dataset):
    def __init__(self, feature_path, labels, split, cls_type, pad_idx, csv_path=None, context_size=512):
        self.pt_dir = feature_path
        self.labels = labels
        self.split = split
        self.csv_path = csv_path  # Path to the CSV file with mood labels
        self.get_fl()
        self.cls_type = cls_type
        self.pad_idx = pad_idx
        self.context_size = context_size  # Maximum sequence length to keep

        # Mood names mapping for consistent ordering
        self.mood_names = ModelConstants.MOOD_NAMES

        # Create mapping from mood name to index
        self.mood_to_idx = {mood: idx for idx, mood in enumerate(self.mood_names)}

    def get_fl(self):
        # Original code for valence-arousal data
        if self.split == "TRAIN":
            self.fl = pd.read_csv(os.path.join(self.csv_path, "train.csv"), index_col=0)
        elif self.split == "VALID":
            self.fl = pd.read_csv(os.path.join(self.csv_path, "val.csv"), index_col=0)
        elif self.split == "TEST":
            self.fl = pd.read_csv(os.path.join(self.csv_path, "test.csv"), index_col=0)
        # if "file" in self.fl.columns:
        #     self.fl = self.fl.set_index("file")
        else:
            print("Split should be one of [TRAIN, VALID, TEST]")

    def parse_mood_tokens(self, mood_tokens_str):
        """
        Parse a string of mood tokens in format 'Mood_Type_Probability' into a tensor
        of 9 probabilities corresponding to each mood category.
        """
        # Initialize tensor with zeros for all 9 moods
        mood_probs = np.zeros(len(self.mood_names))

        # Split the string into individual mood tokens
        if isinstance(mood_tokens_str, str):
            mood_tokens = mood_tokens_str.split()

            for token in mood_tokens:
                # Parse tokens in format: Mood_Type_Probability
                parts = token.split("_")
                if len(parts) == 3 and parts[0] == "Mood":
                    mood_type = parts[1].upper() + "_KEY"  # Convert to our format (e.g., HAPPY_KEY)
                    probability = float(parts[2])

                    # Set the probability for this mood if it exists in our mapping
                    if mood_type in self.mood_to_idx:
                        mood_probs[self.mood_to_idx[mood_type]] = probability

            # Ensure probabilities sum to 1
            if np.sum(mood_probs) > 0:
                mood_probs = mood_probs / np.sum(mood_probs)

        return torch.tensor(mood_probs, dtype=torch.float)

    def __getitem__(self, index):
        audio_fname = self.fl.index[index]  # Get the file name from index

        # New code for mood vectors from CSV
        if self.cls_type == "MOOD":
            if "mood_tokens" in self.fl.columns:
                # Get mood tokens string from the CSV
                mood_tokens_str = self.fl.iloc[index]["mood_tokens"]
                # Parse mood tokens into probabilities tensor
                labels = self.parse_mood_tokens(mood_tokens_str)
            else:
                # Fallback if mood_tokens column doesn't exist
                print(f"Warning: 'mood_tokens' column not found for {audio_fname}")
                labels = torch.zeros(len(self.mood_names), dtype=torch.float)
        else:
            # Original valence-arousal code
            label = self.fl.iloc[index]["label"]
            if self.cls_type == "AV":
                labels = self.labels.index(label)
            elif self.cls_type == "A":
                if label in ["Q1", "Q2"]:
                    labels = self.labels.index("HA")
                elif label in ["Q3", "Q4"]:
                    labels = self.labels.index("LA")
            elif self.cls_type == "V":
                if label in ["Q1", "Q4"]:
                    labels = self.labels.index("HV")
                elif label in ["Q2", "Q3"]:
                    labels = self.labels.index("LV")

        # Remove .mid extension if present for loading the PT file
        if audio_fname.endswith(".mid"):
            audio_fname = audio_fname[:-4]

        try:
            processed_midi = torch.load(os.path.join(self.pt_dir, audio_fname + ".pt"))
        except FileNotFoundError:
            # Return empty tensor as a placeholder when file is not found
            print(f"Warning: File not found for {audio_fname}, returning empty tensor")
            processed_midi = torch.zeros(1, dtype=torch.long)  # Create minimal placeholder tensor

        # Limit context size if specified
        if self.context_size:
            processed_midi = processed_midi[: self.context_size]

        return processed_midi, labels, audio_fname

    def __len__(self):
        return len(self.fl)

    def batch_padding(self, data):
        texts, labels, audio_fname = list(zip(*data))

        # Check if inputs are tensors and convert appropriately
        is_tensor = isinstance(texts[0], torch.Tensor)

        if is_tensor:
            # Tensor-based padding approach
            max_len = max([len(s) for s in texts])
            padded_texts = []

            for s in texts:
                if len(s) < max_len:
                    # Create padding
                    padding = torch.full((max_len - len(s),), self.pad_idx, dtype=s.dtype)
                    # Concatenate the tensor with padding
                    padded_s = torch.cat([s, padding])
                else:
                    padded_s = s
                padded_texts.append(padded_s)

            # Stack all padded tensors
            batch_texts = torch.stack(padded_texts)
        else:
            # Original list-based approach
            max_len = max([len(s) for s in texts])
            batch_texts = torch.LongTensor([s + [self.pad_idx] * (max_len - len(s)) if len(s) < max_len else s for s in texts])

        # Handle mood vector labels if present
        if self.cls_type == "MOOD":
            return batch_texts, torch.stack(labels), audio_fname
        else:
            # Original code for valence-arousal
            return batch_texts, torch.LongTensor(labels), audio_fname
