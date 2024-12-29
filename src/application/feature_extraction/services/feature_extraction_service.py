import asyncio
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

import pretty_midi as pm

from sklearn.manifold import TSNE

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.persistence.dataloader.repositories.dataloader_repository import (
    save_async,
    async_load,
)
from src.application.encoder.models.item_model import Item
from src.application.encoder.helpers.encoder_helper import (
    read_note_tempo,
    extract_downbeats,
    extract_beats,
    quantize_midi,
    group_items,
    extract_dominant_keys,
)
from src.application.feature_extraction.helpers.description_helper import (
    get_description,
)
from src.application.dataloader.models.dataloader_model import DataloaderModule
from src.application.feature_extraction.helpers.latent_features_helper import (
    load_vae_from_checkpoint,
    read_features,
    read_labels,
    plot_reduced_data,
    cluster_reduced_data,
    rank_correlation,
    plot_correlation,
    model_using_ann,
)
from src.domain.models.feature_extraction.feature_extraction_model import (
    VQVAEParameters,
    TSNEParameters,
    CorrelationParameters,
    LatentRepresentationDatasetParameters,
    DescriptionParameters,
    DescriptionDatasetParameters,
)
from src.application.feature_extraction.models.vae_model import VqVaeModule

from src.domain.constants.paths_constants import (
    CHECKPOINTS_PATH,
    VAE_PATH,
    DATALOADER_PATH,
    MIDI_PATH,
    CHORDS_PATH,
    KEYS_PATH,
    DESCRIPTIONS_PATH,
)


def extract_description_sample(parameters: DescriptionParameters):
    if isinstance(parameters.midi, str):
        midi = pm.PrettyMIDI(parameters.midi)
    else:
        midi = parameters.midi

    note_items, tempo_items = read_note_tempo(midi)
    quantize_midi(midi, note_items, midi.resolution)
    beats = extract_beats(midi)
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
    description = get_description(midi, groups)

    sample = {"description": description}

    if parameters.save:
        # Asynchronous saving function
        save_async(parameters.description_out_dir, parameters.midi, sample, "description")

    # return True  # {"Message": "Extracted Description"}


async def extract_description_dataset(parameters: DescriptionDatasetParameters) -> dict:
    dataset_path = os.path.join(MIDI_PATH, parameters.dataset_name)
    chords_out_dir = os.path.join(CHORDS_PATH, parameters.dataset_name)
    keys_out_dir = os.path.join(KEYS_PATH, parameters.dataset_name)
    description_out_dir = os.path.join(DESCRIPTIONS_PATH, parameters.dataset_name)

    async def process_file(file_path: str) -> tuple[bool, str]:
        extract_description_sample(
            DescriptionParameters(
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
    processed = 0

    # Process files in batches
    batch_size = 10
    while processed < total_files:
        batch = midi_files[processed : processed + batch_size]
        batch_tasks = [process_file(os.path.join(dataset_path, file)) for file in batch]

        # Process batch
        results = await asyncio.gather(*batch_tasks, return_exceptions=False)
        processed += len(batch)

    return {"Message": "Extracted Description Dataset"}


def train_vae(parameters: VQVAEParameters) -> dict:
    datamodule_parameters = pickle.load(
        open(
            os.path.join(
                DATALOADER_PATH,
                parameters.dataset_name,
                f"{parameters.dataset_name}_datamodule_parameters.pkl",
            ),
            "rb",
        )
    )

    datamodule_parameters["load_latent"] = False
    datamodule_parameters["load_desc"] = False

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
    datamodule_parameters = pickle.load(
        open(
            os.path.join(
                DATALOADER_PATH,
                parameters.dataset_name,
                f"{parameters.dataset_name}_datamodule_parameters.pkl",
            ),
            "rb",
        )
    )
    datamodule_parameters["load_latent"] = False
    datamodule_parameters["load_desc"] = False
    datamodule_parameters["context_size"] = -1

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


def generate_tsne(parameters: TSNEParameters):
    mat = read_features(parameters.dataset_name)
    mat = np.vstack(mat)
    tsne_reduced_vectors = TSNE(
        n_components=parameters.n_components,
        perplexity=parameters.perplexity,
        early_exaggeration=parameters.early_exaggeration,
        learning_rate=parameters.learning_rate,
        n_iter=parameters.n_iter,
        n_iter_without_progress=parameters.n_iter_without_progress,
        min_grad_norm=parameters.min_grad_norm,
        metric=parameters.metric,
        metric_params=parameters.metric_params,
        init=parameters.init,
        verbose=parameters.verbose,
        random_state=parameters.random_state,
        method=parameters.method,
        angle=parameters.angle,
        n_jobs=parameters.n_jobs,
    ).fit_transform(mat)

    if parameters.cluster:
        labels, label_columns = read_labels(parameters.dataset_name)
        data, label_vectors, cntr = cluster_reduced_data(tsne_reduced_vectors, labels)
        if parameters.plot:
            plot_reduced_data(data, cntr)

    return {"Message": "Generated TSNE"}


def get_correlation(parameters: CorrelationParameters):
    labels, label_columns = read_labels(parameters.dataset_name)
    features = read_features(parameters.dataset_name)
    correlation_matrix = {}
    for label in labels.loc[:, label_columns]:
        y = labels.loc[:, label_columns][label]
        correlation_matrix[label] = []
        for feature, X in features.loc[:, features.columns != "file_name"].items():
            if parameters.correlation_type == "spearmann":
                corr_coeff, p_value = spearmanr(X, y)
            elif parameters.correlation_type == "pearson":
                corr_coeff, p_value = pearsonr(X, y)
            correlation_matrix[label].append(corr_coeff)
    correlation_matrix = pd.DataFrame(correlation_matrix, columns=labels)
    if parameters.rank:
        rank_correlation(correlation_matrix)
    if parameters.plot:
        plot_correlation(correlation_matrix)
    return {"Message": "Got Correlation"}
