import asyncio
import os
import json
import pandas as pd


import torch

import pretty_midi as pm

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from persistence.dataloader.repositories.dataloader_repository import (
    save_async,
    async_load,
)
from application.encoder.helpers.encoder_helper import (
    read_note_tempo,
    extract_downbeats,
    quantize_midi,
    group_items,
    extract_dominant_keys,
)
from application.feature_extraction.helpers.symbolic_features_helper import (
    get_symbolic_features,
    get_piece_level_symbolic_features,
)
from application.dataloader.models.dataloader_model import DataloaderModule
from application.feature_extraction.helpers.latent_features_helper import (
    load_vae_from_checkpoint,
)
from domain.models.feature_extraction.feature_extraction_model import SymbolicFeaturesParameters
from application.feature_extraction.models.vae_model import VqVaeModule

from domain.constants.paths_constants import (
    CHECKPOINTS_PATH,
    VAE_PATH,
    MIDI_PATH,
    PROCESSED_PATH,
    LABELS_PATH,
)


def extract_symbolic_features(parameters: SymbolicFeaturesParameters):
    if isinstance(parameters.midi, str):
        midi = pm.PrettyMIDI(parameters.midi)
    else:
        midi = parameters.midi

    # Try to load processed data if it exists
    try:
        processed_data = async_load(parameters.processed_dir, parameters.midi, "processed")
    except:
        processed_data = {}

    note_items, tempo_items = read_note_tempo(midi)
    quantize_midi(midi, note_items, midi.resolution)
    downbeats = extract_downbeats(midi)

    # Get chords and keys from processed data
    if "chords" not in processed_data or "keys" not in processed_data:
        raise ValueError("Processed data must contain chords and keys. Please run encoder first.")

    remi_chords = processed_data["chords"]["remi_chords"]
    remi_keys = processed_data["keys"]["remi_keys"]

    midi.tonal_plan = remi_keys

    items = remi_keys + remi_chords + tempo_items + note_items
    groups = group_items(midi, downbeats, items=items)
    groups = extract_dominant_keys(groups)

    # Extract features based on level
    if parameters.level == "piece":
        global_features = get_piece_level_symbolic_features(midi, groups)
        
        # For piece-level features, return additional info for CSV saving
        file_name = os.path.basename(parameters.midi) if isinstance(parameters.midi, str) else os.path.basename(processed_data.get("original_file", "unknown.mid"))

        return {"Message": "Global Features Extracted Successfully", "csv_data": {"file": file_name, "global_features": global_features}}
    else:  # default to bar level
        symbolic_features = get_symbolic_features(midi, groups, add_position_tokens=parameters.add_position_tokens)
        
        # Add Symbolic Features to processed data
        if "symbolic_features" not in processed_data:
            processed_data["symbolic_features"] = {}
        processed_data["symbolic_features"] = {f"{parameters.level}_symbolic": symbolic_features}

        # Save to processed data file
        if parameters.save:
            save_async(parameters.processed_dir, parameters.midi, processed_data, "processed")

    return {"Message": "Symbolic Features Extracted Successfully"}


async def extract_symbolic_features_dataset(dataset_name: str = "", level: str = "bar", add_position_tokens: bool = False) -> dict:
    dataset_path = MIDI_PATH
    processed_dir = os.path.join(PROCESSED_PATH, dataset_name)
    
    # Initialize variables for CSV handling if needed
    is_piece_level = level == "piece"
    
    if is_piece_level:        
        # CSV file path
        csv_file_path = os.path.join(LABELS_PATH, f"{dataset_name}.csv")
        
        # Initialize or load the CSV file
        df = pd.read_csv(csv_file_path)


    async def process_file(file_path: str) -> tuple[bool, str]:
        result = extract_symbolic_features(
            SymbolicFeaturesParameters(
                midi=file_path,
                processed_dir=processed_dir,
                save=True,
                level=level,
                add_position_tokens=add_position_tokens,
            )
        )
        
        # If it's piece-level, update CSV file immediately
        if is_piece_level and "csv_data" in result:
            # Get the data from the result
            csv_data = result["csv_data"]
            file_name = csv_data["file"]
            global_features = csv_data["global_features"]
            
            # Update each column individually
            row_idx = df.index[df['file'] == file_name].tolist()[0]
            
            # Add the global_features column if it doesn't exist
            if 'global_features' not in df.columns:
                df['global_features'] = None
            
            # Format global_features as a space-separated string similar to mood_tokens
            if isinstance(global_features, list):
                formatted_features = " ".join(global_features)
            else:
                formatted_features = str(global_features)
            
            df.at[row_idx, 'global_features'] = formatted_features
            
            # Save to CSV file after each update
            df.to_csv(csv_file_path, index=False)
            print(f"Updated {dataset_name}.csv for file {file_name} with global features", flush=True)
        
        return True, ""

    # Create tasks for each file
    all_midi_files = [f for f in os.listdir(dataset_path) if f.endswith((".mid", ".midi"))]
    # Filter to only include files that are in the CSV
    midi_files = [f for f in all_midi_files if f in df['file'].values]
    total_files = len(midi_files)
    print(f"Total files to process: {total_files}", flush=True)
    processed = 0
    successful = 0
    errors = []

    # Process files in batches
    batch_size = 10
    while processed < total_files:
        batch = midi_files[processed : processed + batch_size]
        batch_tasks = [process_file(os.path.join(dataset_path, file)) for file in batch]

        # Process batch
        results = await asyncio.gather(*batch_tasks, return_exceptions=False)

        # Update counters
        for success, error_msg in results:
            if success:
                successful += 1
            else:
                errors.append(error_msg)

        processed += len(batch)

    # Prepare summary message
    position_tokens_str = " with position tokens" if add_position_tokens and level == "bar" else ""
    summary = (
        f"Symbolic Features Extraction Complete ({level} level{position_tokens_str})\n"
        f"Total files: {total_files}\n"
        f"Successfully processed: {successful}\n"
        f"Failed: {len(errors)}\n"
    )
    if errors:
        summary += "\nErrors:\n" + "\n".join(errors)

    return {"Message": summary}


def train_vae(config_path: str) -> dict:
    """Train a VAE model using parameters from the config file."""
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    vae_config = config.get("vae", {})
    datamodule_parameters = config["dataloader"]
    datamodule_parameters["load_latent"] = False
    datamodule_parameters["load_symb"] = False

    datamodule = DataloaderModule(**datamodule_parameters)

    accumulate_grad_batches = vae_config.get("target_batch_size", 256) // datamodule.batch_size
    if vae_config.get("load_from_checkpoint", False):
        model = load_vae_from_checkpoint(vae_config.get("checkpoint_path"))
    else:
        model = VqVaeModule(
            dataset_name=config.get("dataloader", {}).get("dataset_name"),
            d_model=vae_config.get("d_model", 512),
            context_size=datamodule.context_size,
            n_codes=vae_config.get("n_codes", 2048),
            n_groups=vae_config.get("n_groups", 16),
            d_latent=vae_config.get("d_latent", 1024),
            lr=vae_config.get("lr", 1e-4),
            lr_schedule=vae_config.get("lr_schedule", "const"),
            warmup_steps=vae_config.get("warmup_steps", 4000),
            max_steps=vae_config.get("max_steps", 100000000000000000000),
            encoder_layers=vae_config.get("encoder_layers", 4),
            decoder_layers=vae_config.get("decoder_layers", 6),
            encoder_ffn_dim=vae_config.get("encoder_ffn_dim", 2048),
            decoder_ffn_dim=vae_config.get("decoder_ffn_dim", 2048),
            windowed_attention_pr=vae_config.get("windowed_attention_pr", 0.0),
            max_lookahead=vae_config.get("max_lookahead", 4),
            disable_vq=vae_config.get("disable_vq", False),
            accumulate_grad_batches=accumulate_grad_batches,
            max_positions=datamodule.max_positions,
            automatic_optimization=vae_config.get("automatic_optimization", False),
            beta=vae_config.get("beta", 0.02),
            cycle_length=vae_config.get("cycle_length", 2000),
            position_embedding_type=vae_config.get("position_embedding_type", "relative_key_query"),
            num_attention_heads=vae_config.get("num_attention_heads", 8),
            decay=vae_config.get("decay", 0.995),
            eps=vae_config.get("eps", 1e-4),
            restart_threshold=vae_config.get("restart_threshold", 0.99),
        )

    device = torch.device(vae_config.get("device", "cuda"))
    model.to(device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()
    checkpoint_path = os.path.join(CHECKPOINTS_PATH, config.get("dataloader", {}).get("dataset_name"), vae_config.get("training_name"), "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=checkpoint_path,
        filename="{step}-{valid_loss:.2f}",
        save_last=True,
        save_top_k=0,
        every_n_train_steps=500,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        default_root_dir=os.path.join(CHECKPOINTS_PATH, config.get("dataloader", {}).get("dataset_name"), vae_config.get("training_name"), "training_logs"),
        devices=device_count,
        accelerator="gpu",
        profiler="simple",
        callbacks=[checkpoint_callback, lr_monitor],
        enable_checkpointing=True,
        max_epochs=vae_config.get("epochs", 100),
        max_steps=vae_config.get("max_training_steps", 100000),
        log_every_n_steps=max(100, min(25 * accumulate_grad_batches, 200)),
        val_check_interval=max(500, min(300 * accumulate_grad_batches, 1000)),
        limit_val_batches=64,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)

    return {"Message": "Trained VAE"}


def generate_latent_representations_dataset(
    config_path: str,
) -> dict:
    """Generate latent representations using parameters from the config file."""
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    vae_config = config.get("vae", {})
    datamodule_parameters = config["dataloader"]

    datamodule_parameters["encode"] = False
    datamodule_parameters["load_latent"] = False
    datamodule_parameters["load_symb"] = False
    datamodule_parameters["context_size"] = -1

    # Convert train_val_test_split from list to tuple if needed
    if isinstance(datamodule_parameters["train_val_test_split"], list):
        datamodule_parameters["train_val_test_split"] = tuple(datamodule_parameters["train_val_test_split"])

    datamodule = DataloaderModule(**datamodule_parameters)

    accumulate_grad_batches = vae_config.get("target_batch_size", 256) // datamodule.batch_size
    checkpoint_path = config.get("vae", {}).get("checkpoint_path")

    model = load_vae_from_checkpoint(checkpoint_path)

    device = torch.device(vae_config.get("device", "cuda"))
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} does not exist")

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=checkpoint_path,
        filename="{step}-{valid_loss:.2f}",
        save_last=True,
        save_top_k=2,
        every_n_train_steps=100,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        default_root_dir=os.path.join(CHECKPOINTS_PATH, datamodule_parameters.get("dataset_name"), vae_config.get("training_name"), "training_logs"),
        devices=device_count,
        accelerator="gpu",
        profiler="simple",
        callbacks=[checkpoint_callback, lr_monitor],
        enable_checkpointing=True,
        max_epochs=vae_config.get("epochs", 100),
        max_steps=vae_config.get("max_training_steps", 100000),
        log_every_n_steps=max(100, min(25 * accumulate_grad_batches, 200)),
        val_check_interval=max(500, min(300 * accumulate_grad_batches, 1000)),
        limit_val_batches=64,
        num_sanity_val_steps=0,
    )

    predictions = trainer.predict(model, datamodule=datamodule)
    return {"Message": "Generated Latent Representations"}
