import os
import pickle
import json
import glob
import tempfile
import shutil
import asyncio
from typing import Dict, Any, Union

import pandas as pd
from tqdm import tqdm
import copy

import torch
import torch.multiprocessing
from application.dataloader.helper.dataloader_helper import represent_encoding
from application.encoder.helpers.remi_helper import remi2midi

from application.generator.models.generator_model import MIDIGeneratorModule
from domain.models.generator_model import GenerateFromMIDIParameters, GeneratorTrainingParameters

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback

from application.dataloader.models.dataloader_model import (
    DataloaderModule,
)
from application.generator.helper.generator_helpers import (
    load_generator_from_checkpoint,
    load_generator_weights,
)

from domain.constants.paths_constants import (
    CHECKPOINTS_PATH,
    GENERATED_PATH,
    PROCESSED_PATH,
    LATENTS_PATH,
    MIDI_PATH,
)
from domain.constants.model_constants import ModelConstants
from persistence.dataloader.repositories.dataloader_repository import async_load, CPU_Unpickler, save_async


class WeightsOnlyCheckpoint(ModelCheckpoint):
    """Custom checkpoint callback that saves only weights and hyperparameters."""

    def __init__(self, dirpath, filename, monitor, save_last=True, save_top_k=0, every_n_train_steps=None):
        super().__init__(
            dirpath=dirpath, filename=filename, monitor=monitor, save_last=save_last, save_top_k=save_top_k, every_n_train_steps=every_n_train_steps
        )
        self.hyperparams_saved = False

    def _save_checkpoint(self, trainer, filepath):
        # Extract model hyperparameters
        hyperparameters = trainer.lightning_module.hparams

        # Extract state dict and remove position_ids if present
        state_dict = trainer.lightning_module.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith("embeddings.position_ids")}

        # Save weights
        weights_path = filepath.replace(".ckpt", "_weights.pt")
        torch.save(state_dict, weights_path)

        # Save hyperparameters only once
        if not self.hyperparams_saved:
            config_path = os.path.join(os.path.dirname(filepath), "model_config.json")
            with open(config_path, "w") as f:
                json.dump(hyperparameters, f, indent=2)
            self.hyperparams_saved = True

        # For compatibility, also save a small marker file at the original path
        checkpoint_marker = {"weights_path": weights_path, "epoch": trainer.current_epoch, "global_step": trainer.global_step}
        torch.save(checkpoint_marker, filepath)


def train_generator(parameters: GeneratorTrainingParameters) -> dict:
    """Train a generator model using BaseModel parameters."""
    torch.multiprocessing.set_start_method("spawn")

    # Set up data module parameters
    datamodule_parameters = {
        "dataset_name": parameters.dataset_name,
        "context_size": parameters.context_size,
        "max_positions": parameters.max_positions,
        "max_bars": parameters.max_bars,
        "max_bars_per_context": parameters.max_bars_per_context,
        "max_contexts_per_file": parameters.max_contexts_per_file,
        "bar_token_mask": parameters.bar_token_mask,
        "bar_token_idx": parameters.bar_token_idx,
        "batch_size": parameters.batch_size,
        "num_workers": parameters.num_workers,
        "pin_memory": parameters.pin_memory,
        "train_val_test_split": parameters.train_val_test_split,
        "load_latent": parameters.load_latent,
        "load_symb": parameters.load_symb,
        "load_emotions": parameters.load_emotions,
        "load_global_features": parameters.load_global_features,
        "load_text_prompts": parameters.load_text_prompts,
        "encode": parameters.encode,
        "caption": parameters.caption,
    }

    datamodule = DataloaderModule(**datamodule_parameters)

    accumulate_grad_batches = parameters.batch_size // datamodule.batch_size
    
    if parameters.load_from_checkpoint and parameters.checkpoint_path:
        # Load from checkpoint
        model = load_generator_from_checkpoint(parameters.checkpoint_path, eval=False)
    elif parameters.load_weights and parameters.weights_path and parameters.config_path:
        # Load from separate weights and config files
        model = load_generator_weights(parameters.weights_path, parameters.config_path, eval=False)
    else:
        # Create new model from parameters
        model_params = {
            "d_model": parameters.d_model,
            "d_latent": parameters.d_latent,
            "context_size": parameters.context_size,
            "max_bars": parameters.max_bars,
            "max_positions": parameters.max_positions,
            "lr": parameters.lr,
            "lr_schedule": parameters.lr_schedule,
            "warmup_steps": parameters.warmup_steps,
            "max_steps": parameters.max_steps,
            "encoder_layers": parameters.encoder_layers,
            "decoder_layers": parameters.decoder_layers,
            "intermediate_size": parameters.intermediate_size,
            "num_attention_heads": parameters.num_attention_heads,
            "device": parameters.device,
        }
        
        model = MIDIGeneratorModule(**model_params)

    device = torch.device(parameters.device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(parameters.checkpoint_dir, exist_ok=True)

    checkpoint_callback = WeightsOnlyCheckpoint(
        dirpath=parameters.checkpoint_dir,
        filename="{step}-{val_loss:.2f}",
        monitor="val_loss",
        save_last=True,
        save_top_k=parameters.save_top_k,
        every_n_train_steps=parameters.every_n_train_steps,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        max_steps=parameters.max_steps,
        max_epochs=parameters.max_epochs,
        accelerator="gpu" if device_count > 0 else "cpu",
        devices=min(device_count, parameters.gpus) if device_count > 0 else "auto",
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=parameters.val_check_interval,
        log_every_n_steps=parameters.log_every_n_steps,
        limit_val_batches=parameters.limit_val_batches,
        num_sanity_val_steps=parameters.num_sanity_val_steps,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    try:
        trainer.fit(model, datamodule=datamodule)
        return {"Message": "Generator training completed successfully"}
    except Exception as e:
        return {"Message": f"Generator training failed: {str(e)}"}


def generate_from_midi(parameters: GenerateFromMIDIParameters, model=None) -> dict:
    """Generate a sample using existing MIDI files as prompts."""

    # Load the model from checkpoint if not provided
    if model is None:
        # # Load the model from checkpoint
        weights_path = parameters.weights_path
        training_config_path = parameters.config_path
        model = load_generator_weights(weights_path, training_config_path)

        # # Original loading method
        # model = load_generator_from_checkpoint(parameters.checkpoint_path)

        model = model.to("cuda")

    processed_dir = os.path.join(PROCESSED_PATH, "ReMIDICaps")

    # Load latents
    latent_data = async_load(processed_dir, parameters.latent_midi, "processed")
    latents_path = os.path.join(
        str(LATENTS_PATH),
        "ReMIDICaps",
        f"{os.path.basename(parameters.latent_midi)}_latents.pkl",
    )
    latents_file = CPU_Unpickler(open(latents_path, "rb")).load()

    # Load symbolic and emotions data
    symbolic_data = async_load(processed_dir, parameters.symbolic_midi, "processed")
    # emotions_data = async_load(processed_dir, parameters.emotions_midi, "processed")

    # Prepare batch for inference
    latent_rep = represent_encoding(
        parameters.latent_midi,
        "",
        parameters.context_size,
        parameters.max_bars,
        parameters.max_positions,
        None,
        -1,
        -1,
        events=latent_data["encodings"]["events"],
        latents=latents_file["latents"],
        codes=latents_file["codes"],
        bar_symbolic=None,
        save=False,
    )
    latents = latent_rep["latents"]

    symb_rep = represent_encoding(
        parameters.symbolic_midi,
        "",
        parameters.context_size,
        parameters.max_bars,
        parameters.max_positions,
        None,
        -1,
        -1,
        events=symbolic_data["encodings"]["events"],
        latents=None,
        codes=None,
        bar_symbolic=symbolic_data["symbolic_features"]["bar_symbolic"],
        save=False,
    )
    symb_bar_ids = symb_rep["symb_bar_ids"]
    symb_ids = symb_rep["bar_symbolic"]

    representation = {
        "latents": latents,
        "bar_symbolic": symb_ids,
        "symb_bar_ids": symb_bar_ids,
    }

    batch = {key: tensor.unsqueeze(0)[:, : parameters.initial_context] for key, tensor in representation.items()}  # parameters.initial_context

    # Generate the sample
    sample = model.sample(batch, max_length=parameters.max_n_tokens, max_bars=parameters.max_bars, temp=parameters.temperature)

    xs_hat = sample["sequences"].detach().cpu()
    events_hat = [model.vocab.decode(x) for x in xs_hat]

    try:
        pm_hat = remi2midi(events_hat[0])
    except Exception as err:
        return {"Message": f"ERROR: Could not convert events to midi: {err}"}

    # Create output directory if it doesn't exist
    output_dir = os.path.join(GENERATED_PATH, parameters.output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the generated MIDI
    output_path = os.path.join(output_dir, f"{parameters.output_name}.mid")
    pm_hat.write(output_path)

    return {"Message": "Generated Sample", "output_path": output_path}


def save_checkpoint_separate(checkpoint_path: str, output_dir: str) -> dict:
    """
    Load a model checkpoint (.pkl or .ckpt) and save weights as .pt and hyperparameters as .json

    Args:
        checkpoint_path: Path to the original checkpoint file
        output_dir: Directory to save the separate files

    Returns:
        Dictionary with paths to the saved files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the original checkpoint
    pl_ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Extract hyperparameters and state dict
    hyperparameters = pl_ckpt["hyper_parameters"]
    state_dict = pl_ckpt["state_dict"]

    # Filter out position_ids (same as in the original loader)
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith("embeddings.position_ids")}

    # Generate filenames (using the original filename as a base)
    base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    weights_path = os.path.join(output_dir, f"{base_name}_weights.pt")
    config_path = os.path.join(output_dir, f"{base_name}_config.json")

    # Save the files
    torch.save(state_dict, weights_path)

    with open(config_path, "w") as f:
        json.dump(hyperparameters, f, indent=2)

    return {"weights_path": weights_path, "config_path": config_path, "message": "Checkpoint successfully separated"}


def batch_generate_from_dataset(dataset_csv_path: str, parameters: GenerateFromMIDIParameters) -> dict:
    """
    Generate a batch of MIDI files from a CSV dataset.

    Args:
        dataset_csv_path: Path to the CSV file containing MIDI file information
        parameters: GenerateFromMIDIParameters object containing generation parameters

    Returns:
        Dictionary with message about batch generation and output paths
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join(GENERATED_PATH, parameters.output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the CSV dataset
    print(f"Loading dataset from {dataset_csv_path}")
    df = pd.read_csv(dataset_csv_path)

    # Load the model only once for reuse
    if parameters.load_from_checkpoint:
        generator_checkpoint = parameters.checkpoint_path
        model = load_generator_from_checkpoint(generator_checkpoint, eval=False)
    elif parameters.load_weights:
        weights_path = parameters.weights_path
        training_config_path = parameters.config_path
        model = load_generator_weights(weights_path, training_config_path, eval=False)

    # Move model to GPU
    device = torch.device(parameters.device)
    model = model.to(device)

    batch_results = []

    # Process each MIDI file in the dataset
    print(f"Generating {len(df)} MIDI files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Get filename and mood tokens
        filename = row["file"]
        mood_tokens = row["mood_tokens"].replace(" ", "_")

        parameters.latent_midi = filename
        parameters.symbolic_midi = filename

        # Create output filename using mood tokens and original filename
        output_name = f"{mood_tokens}_{os.path.splitext(filename)[0]}_generated"

        # Create a copy of parameters with the updated output name
        file_params = copy.deepcopy(parameters)
        file_params.output_name = output_name

        try:
            # Use the existing generate_from_midi function with the model
            result = generate_from_midi(file_params, model=model)

            # Record success
            batch_results.append({"filename": filename, "output_path": result.get("output_path", ""), "status": "success"})

        except Exception as err:
            # Record failure
            batch_results.append({"filename": filename, "error": str(err), "status": "failed"})
            print(f"ERROR generating {filename}: {err}")

    # Count successes and failures
    successes = sum(1 for result in batch_results if result["status"] == "success")
    failures = sum(1 for result in batch_results if result["status"] == "failed")

    return {"Message": f"Batch generation complete. Generated {successes} MIDI files, {failures} failures.", "output_dir": output_dir, "results": batch_results}
