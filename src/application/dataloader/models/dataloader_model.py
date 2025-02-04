import glob
import math
import os
import pickle

import torch
from lightning import LightningDataModule

from torch.utils.data import IterableDataset, DataLoader

# from torchdata.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.iter import IterableWrapper

from application.dataloader.helper.dataloader_helper import (
    represent_encoding,
)
from application.dataloader.models.dataloader_seq_collator_model import SeqCollator
from application.emotion_mapper.helpers.emotion_mapper_helper import read_label_for_midi
from application.encoder.models.vocab_model import RemiVocab, SymbolicFeaturesVocab
from domain.constants.encoder.token_constants import PAD_TOKEN
from domain.constants.paths_constants import (
    MIDI_PATH,
    PROCESSED_PATH,
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
        load_latent,
        load_symb,
        load_emotions,
        encode=False,
        caption=False,
    ):
        super().__init__()

        self.midi_files = glob.glob(
            os.path.join(MIDI_PATH, "**/*.mid"),
            recursive=True,
        )
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
            "encode": self.encode,
            "caption": self.caption,
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

        # self.train_ds = torchdata.datapipes.iter.Shuffler(self.train_ds, buffer_size=2048)
        self.train_ds = IterableWrapper(self.train_ds)
        self.train_ds.shuffle(buffer_size=2048)

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
        encode=False,
        caption=False,
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
        self.encode = encode
        self.caption = caption
        self.symb_vocab = SymbolicFeaturesVocab()

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

                if self.encode:
                    if processed_data and "encoding" in processed_data:
                        continue
                    else:
                        # For encoding mode, just yield the file path
                        yield {"file": self.split[i]}
                        continue

                # Skip if no processed data available
                if not processed_data:
                    print(f"No processed data available for {self.split[i]}")
                    continue

                encoding = processed_data["encoding"]

                # Get symbolic features if needed
                bar_symbolic, piece_symbolic = None, None
                if self.load_symb and "symbolic_features" in processed_data:
                    if "bar_symbolic" in processed_data["symbolic_features"]:
                        bar_symbolic = processed_data["symbolic_features"]["bar_symbolic"]
                    if "piece_symbolic" in processed_data["symbolic_features"]:
                        piece_symbolic = processed_data["symbolic_features"]["piece_symbolic"]

                # Get latents if needed
                latents = None
                codes = None
                if self.load_latent and "latents" in processed_data:
                    latents = processed_data["latents"]["latents"]
                    codes = processed_data["latents"]["codes"]

                # Get emotion vector if needed
                piece_emotions_vector, piece_emotions_tokens = None, None
                if self.load_emotions and "emotions" in processed_data:
                    if "piece_emotions_vector" in processed_data["emotions"]["piece_emotions"]:
                        piece_emotions_vector = processed_data["emotions"]["piece_emotions"]["piece_emotions_vector"]
                    if "piece_emotions_tokens" in processed_data["emotions"]["piece_emotions"]:
                        piece_emotions_tokens = processed_data["emotions"]["piece_emotions"]["piece_emotions_tokens"]

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
                    piece_symbolic,
                    piece_emotions_vector,
                    piece_emotions_tokens,
                    save=False,
                )

            except FileNotFoundError as err:
                print(err)
                continue
            except Exception as err:
                print(f"Error loading {os.path.basename(self.split[i])}: {str(err)}")
                continue

            yield x
