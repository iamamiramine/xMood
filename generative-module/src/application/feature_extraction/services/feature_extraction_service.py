import asyncio
import os
import json
import pandas as pd
import time
from typing import Dict, Any, Union

import torch

import pretty_midi as pm

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

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
from application.feature_extraction.models.vae_model import VqVaeModule
from domain.models.feature_extraction.feature_extraction_model import (
    SymbolicFeaturesParameters,
    SymbolicFeaturesDatasetParameters,
    VaeTrainingParameters,
    LatentRepresentationParameters,
)
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
        file_name = (
            os.path.basename(parameters.midi) if isinstance(parameters.midi, str) else os.path.basename(processed_data.get("original_file", "unknown.mid"))
        )

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


async def extract_symbolic_features_dataset(parameters: SymbolicFeaturesDatasetParameters) -> dict:
    """
    Extract symbolic features from an entire dataset using BaseModel parameters.

    Args:
        parameters: SymbolicFeaturesDatasetParameters containing all configuration

    Returns:
        dict: Summary of the extraction process
    """
    
    # Set up paths
    dataset_path = MIDI_PATH
    processed_dir = parameters.processed_dir or os.path.join(PROCESSED_PATH, parameters.dataset_name)
    output_dir = parameters.output_dir or processed_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get MIDI files from the dataset path
    midi_files = [f for f in os.listdir(dataset_path) if f.endswith((".mid", ".midi"))]
    
    # Limit files if specified
    if parameters.max_files:
        midi_files = midi_files[:parameters.max_files]
    
    # Resume from specific file if specified
    if parameters.resume_from:
        try:
            start_index = midi_files.index(parameters.resume_from)
            midi_files = midi_files[start_index:]
        except ValueError:
            print(f"Warning: Resume file {parameters.resume_from} not found, starting from beginning")

    total_files = len(midi_files)
    processed = 0
    successful = 0
    errors = []

    async def process_file(file_path: str) -> tuple[bool, str]:
        try:
            filename = os.path.basename(file_path)
            
            # Check if already processed and not overwriting
            output_file = os.path.join(output_dir, f"{filename}_processed.pkl")
            if os.path.exists(output_file) and not parameters.overwrite_existing:
                return True, f"Skipped {filename} (already exists)"

            # Load processed data
            try:
                processed_data = async_load(processed_dir, filename, "processed")
            except Exception as e:
                return False, f"Could not load processed data for {filename}: {str(e)}"

            # Check if encodings exist
            if "encodings" not in processed_data:
                return False, f"No encodings found for {filename}"

            # Extract symbolic features based on level
            if parameters.level == "piece":
                symbolic_features = get_piece_level_symbolic_features(
                    midi=file_path,
                    groups=processed_data["encodings"].get("groups", []),
                    omit_time_sig=parameters.omit_time_sig,
                    omit_instruments=parameters.omit_instruments,
                    omit_chords=parameters.omit_chords,
                    omit_meta=parameters.omit_meta,
                )
            else:  # bar level
                symbolic_features = get_symbolic_features(
                    midi=file_path,
                    groups=processed_data["encodings"].get("groups", []),
                    omit_time_sig=parameters.omit_time_sig,
                    omit_instruments=parameters.omit_instruments,
                    omit_chords=parameters.omit_chords,
                    omit_meta=parameters.omit_meta,
                    add_position_tokens=parameters.add_position_tokens,
                )

            # Add symbolic features to processed data
            if "symbolic_features" not in processed_data:
                processed_data["symbolic_features"] = {}
            
            if parameters.level == "piece":
                processed_data["symbolic_features"]["piece_symbolic"] = symbolic_features
            else:
                processed_data["symbolic_features"]["bar_symbolic"] = symbolic_features

            # Save updated processed data if requested
            if parameters.save:
                save_async(output_dir, filename, processed_data, "processed")

            return True, ""
        except Exception as e:
            filename = os.path.basename(file_path)
            if parameters.skip_invalid:
                return False, f"Skipped invalid file {filename}: {str(e)}"
            else:
                return False, f"Error processing {filename}: {str(e)}"

    # Process files in batches
    while processed < total_files:
        batch = midi_files[processed : processed + parameters.batch_size]
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
        print(f"Processed {processed}/{total_files} files", flush=True)

    # Prepare summary message
    summary = f"Symbolic Features Extraction Complete\n" f"Total files: {total_files}\n" f"Successfully processed: {successful}\n" f"Failed: {len(errors)}\n"
    if errors:
        summary += "\nErrors:\n" + "\n".join(errors[:10])
        if len(errors) > 10:
            summary += f"\n... and {len(errors) - 10} more errors"

    return {"Message": summary}


def train_vae(parameters: VaeTrainingParameters) -> dict:
    """Train a VAE model using BaseModel parameters."""
    
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
        model = load_vae_from_checkpoint(parameters.checkpoint_path)
    else:
        # Create new model from parameters
        model_params = {
            "dataset_name": parameters.dataset_name,
            "d_model": parameters.d_model,
            "context_size": parameters.context_size,
            "n_codes": parameters.n_codes,
            "n_groups": parameters.n_groups,
            "d_latent": parameters.d_latent,
            "lr": parameters.lr,
            "lr_schedule": parameters.lr_schedule,
            "warmup_steps": parameters.warmup_steps,
            "max_steps": parameters.max_steps,
            "encoder_layers": parameters.encoder_layers,
            "decoder_layers": parameters.decoder_layers,
            "encoder_ffn_dim": parameters.encoder_ffn_dim,
            "decoder_ffn_dim": parameters.decoder_ffn_dim,
            "windowed_attention_pr": parameters.windowed_attention_pr,
            "max_lookahead": parameters.max_lookahead,
            "disable_vq": parameters.disable_vq,
            "accumulate_grad_batches": accumulate_grad_batches,
            "max_positions": parameters.max_positions,
            "automatic_optimization": parameters.automatic_optimization,
            "beta": parameters.beta,
            "cycle_length": parameters.cycle_length,
            "position_embedding_type": parameters.position_embedding_type,
            "num_attention_heads": parameters.num_attention_heads,
            "decay": parameters.decay,
            "eps": parameters.eps,
            "restart_threshold": parameters.restart_threshold,
        }
        
        model = VqVaeModule(**model_params)

    device = torch.device(parameters.device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(parameters.checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=parameters.checkpoint_dir,
        filename="{step}-{val_loss:.2f}",
        save_last=True,
        save_top_k=parameters.save_top_k,
        every_n_train_steps=parameters.every_n_train_steps,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = WandbLogger(project="vae-training") if hasattr(parameters, 'use_wandb') and parameters.use_wandb else None

    trainer = L.Trainer(
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
        logger=logger,
    )

    try:
        trainer.fit(model, datamodule=datamodule)
        return {"Message": "VAE training completed successfully"}
    except Exception as e:
        return {"Message": f"VAE training failed: {str(e)}"}


async def generate_latent_representations_dataset(parameters: LatentRepresentationParameters) -> dict:
    """Generate latent representations using BaseModel parameters."""
    
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
    checkpoint_path = parameters.checkpoint_path

    model = load_vae_from_checkpoint(checkpoint_path)

    device = torch.device(parameters.device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()

    if not os.path.exists(parameters.output_dir):
        os.makedirs(parameters.output_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=parameters.output_dir,
        filename="{step}-{valid_loss:.2f}",
        save_last=True,
        save_top_k=parameters.save_top_k,
        every_n_train_steps=parameters.every_n_train_steps,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        max_steps=parameters.max_steps,
        accelerator="gpu" if device_count > 0 else "cpu",
        devices=min(device_count, 1) if device_count > 0 else "auto",
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=parameters.val_check_interval,
        log_every_n_steps=parameters.log_every_n_steps,
        limit_val_batches=parameters.limit_val_batches,
        num_sanity_val_steps=parameters.num_sanity_val_steps,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    try:
        predictions = trainer.predict(model, datamodule=datamodule)
        
        # Process and save predictions if requested
        if parameters.save_latents or parameters.save_codes:
            processed_predictions = []
            for batch_predictions in predictions:
                for prediction in batch_predictions:
                    if isinstance(prediction, dict):
                        processed_predictions.append(prediction)
            
            # Save predictions to the output directory
            output_file = os.path.join(parameters.output_dir, "latent_representations.json")
            with open(output_file, 'w') as f:
                json.dump(processed_predictions, f, indent=2)
        
        return {"Message": f"Latent representation generation completed. {len(predictions)} batches processed."}
    except Exception as e:
        return {"Message": f"Latent representation generation failed: {str(e)}"}
