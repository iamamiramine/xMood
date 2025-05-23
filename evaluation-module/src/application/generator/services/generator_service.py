import os
import pickle
import json
import glob
import tempfile
import shutil

import pandas as pd
from tqdm import tqdm
import copy

import torch
from application.dataloader.helper.dataloader_helper import represent_encoding
from application.encoder.helpers.remi_helper import remi2midi

from application.generator.models.generator_model import MIDIGeneratorModule
from domain.models.generator_model import GenerateFromMIDIParameters

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
)
from persistence.dataloader.repositories.dataloader_repository import async_load, CPU_Unpickler


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


def train_generator(config_path: str) -> dict:
    """Train a generator model using parameters from the config file."""
    torch.multiprocessing.set_start_method("spawn")

    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    generator_config = config.get("generator", {})
    datamodule_parameters = config["dataloader"]
    datamodule_parameters["encode"] = False

    # Convert train_val_test_split from list to tuple if needed
    if isinstance(datamodule_parameters["train_val_test_split"], list):
        datamodule_parameters["train_val_test_split"] = tuple(datamodule_parameters["train_val_test_split"])

    datamodule = DataloaderModule(**datamodule_parameters)

    accumulate_grad_batches = generator_config.get("target_batch_size", 256) // datamodule.batch_size

    if generator_config["load_from_checkpoint"]:
        generator_checkpoint = generator_config.get("checkpoint_path")
        model = load_generator_from_checkpoint(generator_checkpoint, eval=False)
    elif generator_config["load_weights"]:
        weights_path = generator_config.get("weights_path")
        training_config_path = generator_config.get("config_path")
        model = load_generator_weights(weights_path, training_config_path, eval=False)
    else:
        model = MIDIGeneratorModule(
            d_model=config.get("vae", {}).get("d_model", 512),
            d_latent=config.get("vae", {}).get("d_latent", 1024),
            context_size=datamodule_parameters["context_size"],
            max_bars=datamodule_parameters["max_bars"],
            max_positions=datamodule_parameters["max_positions"],
            lr=generator_config.get("lr", 1e-4),
            lr_schedule=generator_config.get("lr_schedule", "const"),
            warmup_steps=generator_config.get("warmup_steps", 4000),
            max_steps=generator_config.get("max_steps", 100000000000000000000),
            encoder_layers=generator_config.get("encoder_layers", 6),
            decoder_layers=generator_config.get("decoder_layers", 6),
            intermediate_size=generator_config.get("intermediate_size", 2048),
            num_attention_heads=generator_config.get("num_attention_heads", 8),
        )

    device = torch.device(generator_config.get("device", "cuda"))
    model.to(device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()
    checkpoint_path = os.path.join(CHECKPOINTS_PATH, datamodule_parameters.get("dataset_name"), generator_config.get("training_name"), "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Replace standard checkpoint callback with weights-only checkpoint
    checkpoint_callback = WeightsOnlyCheckpoint(
        monitor="valid_loss",
        dirpath=checkpoint_path,
        filename="{step}-{valid_loss:.2f}",
        save_last=True,
        save_top_k=0,
        every_n_train_steps=500,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        default_root_dir=os.path.join(CHECKPOINTS_PATH, datamodule_parameters.get("dataset_name"), generator_config.get("training_name"), "training_logs"),
        devices=device_count,
        accelerator="gpu",
        profiler="simple",
        callbacks=[checkpoint_callback, lr_monitor],
        enable_checkpointing=True,
        max_epochs=generator_config.get("epochs", 100),
        max_steps=generator_config.get("max_training_steps", 100000),
        log_every_n_steps=max(100, min(25 * accumulate_grad_batches, 200)),
        val_check_interval=max(500, min(300 * accumulate_grad_batches, 1000)),
        limit_val_batches=64,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)

    return {"Message": "Trained Generator"}


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
