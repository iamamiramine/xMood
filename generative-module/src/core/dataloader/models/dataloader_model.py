import glob
import math
import os
import pandas as pd

import torch
from lightning import LightningDataModule

from torch.utils.data import IterableDataset, DataLoader

# from torchdata.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.iter import IterableWrapper

from core.dataloader.helpers.dataloader_helper import (
    represent_encoding,
    async_load,
    load_latents_hdf5,
)
from core.dataloader.models.dataloader_seq_collator_model import SeqCollator
from core.symbolic.models.vocab_model import RemiVocab, SymbolicFeaturesVocab, MoodsVocab
from domain.constants.token_constants import PAD_TOKEN
from domain.constants.model_constants import ModelConstants
from domain.constants.paths_constants import (
    MIDI_PATH,
    PROCESSED_PATH,
    LATENTS_PATH,
)


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
    ):
        super().__init__()

        self.midi_files = []

        # Now find the actual full paths of these files
        for ext in ["*.mid", "*.midi"]:
            for f in glob.glob(os.path.join(MIDI_PATH, f"**/{ext}"), recursive=True):
                self.midi_files.append(f)

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
                try:
                    processed_data = async_load(
                        os.path.join(str(PROCESSED_PATH), self.dataset_name),
                        self.split[i],
                        "processed"
                    )
                except FileNotFoundError:
                    print(f"No processed data available for {self.split[i]}")
                    continue

                # Skip if no processed data available
                if not processed_data:
                    print(f"No processed data available for {self.split[i]}")
                    continue

                encoding = processed_data["encodings"]

                # Get symbolic features if needed
                bar_symbolic = None
                if "symbolic_features" in processed_data:
                    if "bar_symbolic" in processed_data["symbolic_features"]:
                        bar_symbolic = processed_data["symbolic_features"]["bar_symbolic"]

                # Get latents if needed
                latents = None
                codes = None
                latents_path = os.path.join(
                    str(LATENTS_PATH),
                    self.dataset_name,
                    f"{os.path.basename((self.split[i]))}_latents.h5",
                )
                if os.path.exists(latents_path):
                    latents_file = load_latents_hdf5(latents_path)
                    latents = latents_file["latents"]
                    codes = latents_file["codes"]

                # Get file basename for CSV lookup
                file_basename = os.path.basename(self.split[i])

                # Helper function to parse space-separated values into a list
                def parse_space_separated(text):
                    if text is None or pd.isna(text) or text == "":
                        return None
                    return text.split()

                # Get or generate representation
                file = os.path.basename(self.split[i])
                events = encoding["events"]

                x = represent_encoding(
                    file,
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

            except FileNotFoundError as err:
                print(err)
                continue
            except Exception as err:
                print(f"Error loading {os.path.basename(self.split[i])}: {str(err)}")
                continue

            yield x
