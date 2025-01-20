import glob
import math
import os
import pickle
import asyncio

import torch
from lightning import LightningDataModule

from torch.utils.data import IterableDataset, DataLoader

# from torchdata.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.iter import IterableWrapper

from src.application.representation.services.representation_services import (
    represent_encoding,
)
from src.persistence.dataloader.repositories.dataloader_repository import CPU_Unpickler
from src.application.dataloader.models.dataloader_seq_collator_model import SeqCollator
from src.application.encoder.models.vocab_model import RemiVocab, SymbolicFeaturesVocab
from src.domain.constants.encoder.token_constants import PAD_TOKEN
from src.domain.constants.paths_constants import (
    ENCODINGS_PATH,
    DATALOADER_PATH,
    LATENTS_PATH,
    MIDI_PATH,
    SYMBOLIC_FEATURES_PATH,
    REPRESENTATIONS_PATH,
    LABELS_PATH,
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
    ):
        super().__init__()

        self.midi_files = glob.glob(
            os.path.join(os.path.join(MIDI_PATH, dataset_name), "**/*.mid"),
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
        self.desc_vocab = SymbolicFeaturesVocab()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.split = _get_split(self.files, worker_info)

        split_len = len(self.split)

        for i in range(split_len):
            try:
                encoding_file = os.path.join(
                    str(ENCODINGS_PATH),
                    self.dataset_name,
                    f"{os.path.basename((self.split[i]))}_encoding.pkl",
                )
                encoding = pickle.load(open(encoding_file, "rb"))
            except FileNotFoundError as err:
                print(err)
                # raise err
                continue
            if self.load_symb:
                try:
                    symb_file = os.path.join(
                        str(SYMBOLIC_FEATURES_PATH),
                        self.dataset_name,
                        f"{os.path.basename((self.split[i]))}_symbolic.pkl",
                    )
                    symb = pickle.load(open(symb_file, "rb"))
                    symbolic = symb["symbolic"]
                except FileNotFoundError as err:
                    print(err)
                    # raise err
                    continue
            else:
                symbolic = None
            if self.load_latent:
                try:
                    latents_path = os.path.join(
                        str(LATENTS_PATH),
                        self.dataset_name,
                        f"{os.path.basename((self.split[i]))}_latents.pkl",
                    )
                    latents_file = CPU_Unpickler(open(latents_path, "rb")).load()
                    latents = latents_file["latents"]
                    codes = latents_file["codes"]
                except FileNotFoundError as err:
                    print(err)
                    # raise err
                    continue
            else:
                latents = None
                codes = None

            file = os.path.basename(self.split[i])
            events = encoding["events"]

            if os.path.isfile(
                os.path.join(
                    REPRESENTATIONS_PATH,
                    self.dataset_name,
                    f"{os.path.basename(file)}_representation.pkl",
                )
            ):
                x = pickle.load(
                    open(
                        os.path.join(
                            REPRESENTATIONS_PATH,
                            self.dataset_name,
                            f"{os.path.basename(file)}_representation.pkl",
                        ),
                        "rb",
                    )
                )
            else:
                if not os.path.exists(os.path.join(REPRESENTATIONS_PATH, self.dataset_name)):
                    os.makedirs(os.path.join(REPRESENTATIONS_PATH, self.dataset_name))
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
                    symbolic,
                    save=True,
                    out_dir=os.path.join(REPRESENTATIONS_PATH, self.dataset_name),
                )

            yield x
