import asyncio
import os
import json

import torch

import pretty_midi as pm

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.persistence.dataloader.repositories.dataloader_repository import (
    save_async,
    async_load,
)
from src.application.encoder.helpers.encoder_helper import (
    read_note_tempo,
    extract_downbeats,
    quantize_midi,
    group_items,
    extract_dominant_keys,
)
from src.application.feature_extraction.helpers.symbolic_features_helper import (
    get_symbolic_features,
)
from src.application.dataloader.models.dataloader_model import DataloaderModule
from src.application.feature_extraction.helpers.latent_features_helper import (
    load_vae_from_checkpoint,
)
from src.domain.models.feature_extraction.feature_extraction_model import (
    VQVAEParameters,
    LatentRepresentationDatasetParameters,
    SymbolicFeaturesParameters,
    SymbolicFeaturesDatasetParameters,
)
from src.application.feature_extraction.models.vae_model import VqVaeModule

from src.domain.constants.paths_constants import (
    CHECKPOINTS_PATH,
    VAE_PATH,
    MIDI_PATH,
    CHORDS_PATH,
    KEYS_PATH,
    SYMBOLIC_FEATURES_PATH,
)


def extract_symbolic_features(parameters: SymbolicFeaturesParameters):
    if isinstance(parameters.midi, str):
        midi = pm.PrettyMIDI(parameters.midi)
    else:
        midi = parameters.midi

    note_items, tempo_items = read_note_tempo(midi)
    quantize_midi(midi, note_items, midi.resolution)
    downbeats = extract_downbeats(midi)

    # Asynchronous file read for chords
    chords = async_load(parameters.chords_out_dir, parameters.midi, "chords")
    remi_chords = chords["remi_chords"]

    # Asynchronous file read for keys
    keys = async_load(parameters.keys_out_dir, parameters.midi, "keys")
    remi_keys = keys["remi_keys"]

    midi.tonal_plan = remi_keys

    items = remi_keys + remi_chords + tempo_items + note_items
    groups = group_items(midi, downbeats, items=items)
    groups = extract_dominant_keys(groups)
    symbolic_features = get_symbolic_features(midi, groups)

    sample = {"symbolic": symbolic_features}

    if parameters.save:
        # Asynchronous saving function
        save_async(parameters.description_out_dir, parameters.midi, sample, "symbolic")

    # return True  # {"Message": "Extracted Symbolic Features"}


async def extract_symbolic_features_dataset(parameters: SymbolicFeaturesDatasetParameters) -> dict:
    dataset_path = os.path.join(MIDI_PATH, parameters.dataset_name)
    chords_out_dir = os.path.join(CHORDS_PATH, parameters.dataset_name)
    keys_out_dir = os.path.join(KEYS_PATH, parameters.dataset_name)
    description_out_dir = os.path.join(SYMBOLIC_FEATURES_PATH, parameters.dataset_name)

    async def process_file(file_path: str) -> tuple[bool, str]:
        extract_symbolic_features(
            SymbolicFeaturesParameters(
                midi=file_path,
                chords_out_dir=chords_out_dir,
                keys_out_dir=keys_out_dir,
                description_out_dir=description_out_dir,
                save=True,
            )
        )
        return True, ""

    # Create tasks for each file
    midi_files = [f for f in os.listdir(dataset_path) if f.endswith((".mid", ".midi"))]
    total_files = len(midi_files)
    print(total_files, flush=True)
    processed = 0

    # Process files in batches
    batch_size = 10
    while processed < total_files:
        batch = midi_files[processed : processed + batch_size]
        batch_tasks = [process_file(os.path.join(dataset_path, file)) for file in batch]

        # Process batch
        results = await asyncio.gather(*batch_tasks, return_exceptions=False)
        processed += len(batch)

    return {"Message": "Extracted Symbolic Features Dataset"}


def train_vae(parameters: VQVAEParameters) -> dict:
    with open("shared/assets/config.json", "r") as f:
        config = json.load(f)

    datamodule_parameters = config["dataloader"]
    datamodule_parameters["load_latent"] = False
    datamodule_parameters["load_symb"] = False

    datamodule_parameters["load_latent"] = False
    datamodule_parameters["load_symb"] = False

    datamodule = DataloaderModule(**datamodule_parameters)

    accumulate_grad_batches = parameters.target_batch_size // datamodule.batch_size
    if parameters.load_from_checkpoint:
        model = load_vae_from_checkpoint(parameters.checkpoint_dir)
    else:
        model = VqVaeModule(
            parameters.dataset_name,
            parameters.d_model,
            datamodule.context_size,
            parameters.n_codes,
            parameters.n_groups,
            parameters.d_latent,
            parameters.lr,
            parameters.lr_schedule,
            parameters.warmup_steps,
            parameters.max_steps,
            parameters.encoder_layers,
            parameters.decoder_layers,
            parameters.encoder_ffn_dim,
            parameters.decoder_ffn_dim,
            parameters.windowed_attention_pr,
            parameters.max_lookahead,
            parameters.disable_vq,
            accumulate_grad_batches,
            datamodule.max_positions,
            parameters.automatic_optimization,
            parameters.beta,
            parameters.cycle_length,
            parameters.position_embedding_type,
            parameters.num_attention_heads,
            parameters.decay,
            parameters.eps,
            parameters.restart_threshold,
        )
    device = torch.device(parameters.device)
    model.to(device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()
    checkpoint_dir = os.path.join(CHECKPOINTS_PATH, parameters.dataset_name, parameters.training_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=checkpoint_dir,
        filename="{step}-{valid_loss:.2f}",
        save_last=True,
        save_top_k=2,
        every_n_train_steps=100,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        default_root_dir=os.path.join(VAE_PATH, parameters.dataset_name, parameters.training_name),
        devices=device_count,
        accelerator="gpu",
        profiler="simple",
        callbacks=[checkpoint_callback, lr_monitor],
        enable_checkpointing=True,
        max_epochs=parameters.epochs,
        max_steps=parameters.max_training_steps,
        log_every_n_steps=max(100, min(25 * accumulate_grad_batches, 200)),
        val_check_interval=max(500, min(300 * accumulate_grad_batches, 1000)),
        limit_val_batches=64,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)

    return {"Message": "Trained VAE"}


def generate_latent_representations_dataset(
    parameters: LatentRepresentationDatasetParameters,
) -> dict:
    with open("shared/assets/config.json", "r") as f:
        config = json.load(f)

    datamodule_parameters = config["dataloader"]
    datamodule_parameters["load_latent"] = False
    datamodule_parameters["load_symb"] = False
    datamodule_parameters["context_size"] = -1

    # Convert train_val_test_split from list to tuple if needed
    if isinstance(datamodule_parameters["train_val_test_split"], list):
        datamodule_parameters["train_val_test_split"] = tuple(datamodule_parameters["train_val_test_split"])

    datamodule = DataloaderModule(**datamodule_parameters)

    accumulate_grad_batches = parameters.target_batch_size // datamodule.batch_size
    vae_checkpoint = os.path.join(
        CHECKPOINTS_PATH,
        parameters.dataset_name,
        parameters.training_name,
        parameters.checkpoint_name,
    )

    model = load_vae_from_checkpoint(vae_checkpoint)

    device = torch.device(parameters.device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()
    checkpoint_dir = os.path.join(CHECKPOINTS_PATH, parameters.dataset_name, parameters.training_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=checkpoint_dir,
        filename="{step}-{valid_loss:.2f}",
        save_last=True,
        save_top_k=2,
        every_n_train_steps=100,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        default_root_dir=os.path.join(VAE_PATH, parameters.dataset_name, parameters.training_name),
        devices=device_count,
        accelerator="gpu",
        profiler="simple",
        callbacks=[checkpoint_callback, lr_monitor],
        enable_checkpointing=True,
        max_epochs=parameters.epochs,
        max_steps=parameters.max_training_steps,
        log_every_n_steps=max(100, min(25 * accumulate_grad_batches, 200)),
        val_check_interval=max(500, min(300 * accumulate_grad_batches, 1000)),
        limit_val_batches=64,
        num_sanity_val_steps=0,
    )

    predictions = trainer.predict(model, datamodule=datamodule)
    return {"Message": "Generated Latent Representations"}
