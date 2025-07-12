import glob
import math
import os
import pickle
import csv
import pandas as pd

import torch
from lightning import LightningDataModule

from torch.utils.data import IterableDataset, DataLoader

# from torchdata.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.iter import IterableWrapper

from application.dataloader.helper.dataloader_helper import (
    represent_encoding,
)
from application.dataloader.models.dataloader_seq_collator_model import SeqCollator
from application.encoder.models.vocab_model import RemiVocab, SymbolicFeaturesVocab, MoodsVocab
from domain.constants.encoder.token_constants import PAD_TOKEN
from domain.constants.model_constants import ModelConstants
from domain.constants.paths_constants import (
    MIDI_PATH,
    PROCESSED_PATH,
    LATENTS_PATH,
    LABELS_PATH,
)
from persistence.dataloader.repositories.dataloader_repository import CPU_Unpickler


def _get_split(files, worker_info):
    if worker_info:
        n_workers = worker_info.num_workers
        worker_id = worker_info.id

        per_worker = math.ceil(len(files) / n_workers)
        start_idx = per_worker * worker_id
        end_idx = start_idx + per_worker

        split = files[start_idx:end_idx]
    else:
        split = files
    return split


class DataloaderModule(LightningDataModule):
    def __init__(
        self,
        dataset_name,
        context_size,  # ALso max_len
        max_positions,
        max_bars,
        max_bars_per_context,
        max_contexts_per_file,
        bar_token_mask,
        bar_token_idx,
        batch_size,
        num_workers,
        pin_memory,
        train_val_test_split,
        load_latent,
        load_symb,
        load_emotions,
        load_global_features=False,  # New parameter
        load_text_prompts=False,  # New parameter
        encode=False,
        caption=False,
    ):
        super().__init__()

        self.midi_files = []

        # Path to the CSV file using LABELS_PATH
        csv_file_path = os.path.join(LABELS_PATH, f"{dataset_name}.csv")

        if os.path.exists(csv_file_path):
            # Read CSV file using pandas
            self.labels_df = pd.read_csv(csv_file_path)
            # Convert file column to a set for faster lookups
            csv_file_names = set(self.labels_df["file"].values)

            # Now find the actual full paths of these files
            for ext in ["*.mid", "*.midi"]:
                for f in glob.glob(os.path.join(MIDI_PATH, f"**/{ext}"), recursive=True):
                    if os.path.basename(f) in csv_file_names:
                        self.midi_files.append(f)

        else:
            raise ValueError(f"CSV file not found: {csv_file_path}")

        self.dataset_name = dataset_name
        self.context_size = context_size
        self.max_positions = max_positions
        self.max_bars = max_bars
        self.max_bars_per_context = max_bars_per_context
        self.max_contexts_per_file = max_contexts_per_file
        self.bar_token_mask = bar_token_mask
        self.bar_token_idx = bar_token_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_test_split = train_val_test_split
        self.load_latent = load_latent
        self.load_symb = load_symb
        self.load_emotions = load_emotions
        self.load_global_features = load_global_features  # Add new parameter
        self.load_text_prompts = load_text_prompts  # Add new parameter
        self.encode = encode
        self.caption = caption
        self.vocab = RemiVocab()

        self.dataset_parameters = {
            "dataset_name": self.dataset_name,
            "context_size": self.context_size,
            "max_positions": self.max_positions,
            "max_bars": self.max_bars,
            "max_bars_per_context": self.max_bars_per_context,
            "max_contexts_per_file": self.max_contexts_per_file,
            "bar_token_mask": self.bar_token_mask,
            "bar_token_idx": self.bar_token_idx,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "train_val_test_split": self.train_val_test_split,
            "vocab": self.vocab,
            "load_latent": self.load_latent,
            "load_symb": self.load_symb,
            "load_emotions": self.load_emotions,
            "load_global_features": self.load_global_features,
            "load_text_prompts": self.load_text_prompts,
            "encode": self.encode,
            "caption": self.caption,
            "labels_df": self.labels_df,
        }

    def setup(self, stage=None):
        n_valid = int(self.train_val_test_split[1] * len(self.midi_files))
        n_test = int(self.train_val_test_split[2] * len(self.midi_files))
        train_files = self.midi_files[n_test + n_valid :]
        valid_files = self.midi_files[n_test : n_test + n_valid]
        test_files = self.midi_files[:n_test]

        predict_files = self.midi_files

        self.train_ds = DataloaderDataset(train_files, **self.dataset_parameters)

        self.valid_ds = DataloaderDataset(valid_files, **self.dataset_parameters)

        self.test_ds = DataloaderDataset(test_files, **self.dataset_parameters)

        self.predict_ds = DataloaderDataset(predict_files, **self.dataset_parameters)

        self.train_ds = IterableWrapper(self.train_ds)
        self.train_ds.shuffle(buffer_size=ModelConstants.DEFAULT_BUFFER_SIZE)

        if self.encode:
            self.collator = None
            self.collator_pred = None
        else:
            self.collator = SeqCollator(pad_token=self.vocab.to_i(PAD_TOKEN), context_size=self.context_size)
            self.collator_pred = SeqCollator(pad_token=self.vocab.to_i(PAD_TOKEN), context_size=-1)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds,
            collate_fn=self.collator_pred,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )


class DataloaderDataset(IterableDataset):
    def __init__(
        self,
        files,
        dataset_name,
        context_size,
        max_positions,
        max_bars,
        max_bars_per_context,
        max_contexts_per_file,
        bar_token_mask,
        bar_token_idx,
        batch_size,
        num_workers,
        pin_memory,
        train_val_test_split,
        vocab,
        load_latent,
        load_symb,
        load_emotions,
        load_global_features=False,
        load_text_prompts=False,
        encode=False,
        caption=False,
        labels_df=None,
    ):
        self.files = files
        self.dataset_name = dataset_name
        self.context_size = context_size
        self.max_positions = max_positions
        self.max_bars = max_bars
        self.max_bars_per_context = max_bars_per_context
        self.max_contexts_per_file = max_contexts_per_file
        self.bar_token_mask = bar_token_mask
        self.bar_token_idx = bar_token_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_test_split = train_val_test_split
        self.vocab = vocab
        self.load_latent = load_latent
        self.load_symb = load_symb
        self.load_emotions = load_emotions
        self.load_global_features = load_global_features
        self.load_text_prompts = load_text_prompts
        self.encode = encode
        self.caption = caption
        self.labels_df = labels_df
        self.symb_vocab = SymbolicFeaturesVocab()
        self.moods_vocab = MoodsVocab()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.split = _get_split(self.files, worker_info)

        split_len = len(self.split)

        for i in range(split_len):
            try:
                processed_data = None
                # Load processed data
                processed_file = os.path.join(
                    str(PROCESSED_PATH),
                    self.dataset_name,
                    f"{os.path.basename((self.split[i]))}_processed.pkl",
                )
                if os.path.exists(processed_file):
                    processed_data = pickle.load(open(processed_file, "rb"))

                # Skip if no processed data available
                if not processed_data:
                    print(f"No processed data available for {self.split[i]}")
                    continue

                encoding = processed_data["encodings"]

                # Get symbolic features if needed
                bar_symbolic = None
                if self.load_symb and "symbolic_features" in processed_data:
                    if "bar_symbolic" in processed_data["symbolic_features"]:
                        bar_symbolic = processed_data["symbolic_features"]["bar_symbolic"]

                # Get latents if needed
                latents = None
                codes = None
                if self.load_latent:
                    latents_path = os.path.join(
                        str(LATENTS_PATH),
                        self.dataset_name,
                        f"{os.path.basename((self.split[i]))}_latents.pkl",
                    )
                    latents_file = CPU_Unpickler(open(latents_path, "rb")).load()
                    latents = latents_file["latents"]
                    codes = latents_file["codes"]

                # Get emotion vector if needed
                moods = None
                global_features = None
                text_prompts = None

                # Get file basename for CSV lookup
                file_basename = os.path.basename(self.split[i])

                # Helper function to parse space-separated values into a list
                def parse_space_separated(text):
                    if text is None or pd.isna(text) or text == "":
                        return None
                    return text.split()

                # Get data from CSV if the file exists in the labels dataframe
                if self.labels_df is not None:
                    # Find the matching row for this file
                    matching_rows = self.labels_df[self.labels_df["file"] == file_basename]

                    if not matching_rows.empty:
                        row = matching_rows.iloc[0]

                        # Extract mood tokens if needed
                        if self.load_emotions and "mood_tokens" in row:
                            # Parse space-separated mood tokens into a list
                            moods = parse_space_separated(row["mood_tokens"])

                        # Extract global features if needed
                        if self.load_global_features and "global_features" in row:
                            # Parse space-separated global features into a list
                            global_features = parse_space_separated(row["global_features"])

                        # Extract text prompts/captions if needed
                        if self.load_text_prompts and "caption" in row:
                            text_prompts = row["caption"]
                    else:
                        print(f"Warning: File {file_basename} not found in labels CSV")
                else:
                    print(f"Warning: No labels dataframe available for {file_basename}")

                # Get or generate representation
                file = os.path.basename(self.split[i])
                events = encoding["events"]

                x = represent_encoding(
                    file,
                    self.dataset_name,
                    self.context_size,
                    self.max_bars,
                    self.max_positions,
                    self.bar_token_mask,
                    self.max_bars_per_context,
                    self.max_contexts_per_file,
                    events,
                    latents,
                    codes,
                    bar_symbolic,
                    save=False,
                )

                # # Add global features and text prompts to the output
                # if global_features is not None:
                #     x["global_features"] = global_features
                #
                # if text_prompts is not None:
                #     x["text_prompts"] = text_prompts
                #
                # if moods is not None:
                #     # moods_ids = torch.tensor(self.moods_vocab.encode(moods), dtype=torch.int)
                #     x["moods"] = moods  # moods  # moods_ids

            except FileNotFoundError as err:
                print(err)
                continue
            except Exception as err:
                print(f"Error loading {os.path.basename(self.split[i])}: {str(err)}")
                continue

            yield x
