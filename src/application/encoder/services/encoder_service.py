# Standard library imports
import asyncio
import os
import json

# PyTorch and Lightning
import torch
from lightning.pytorch import Trainer

# Local application imports - Models
from application.encoder.models.encoder_model import MIDIEncoderModule
from application.dataloader.models.dataloader_model import DataloaderModule
from domain.models.encoder.encoder_model import EncodeParameters

# Constants - File paths
from domain.constants.paths_constants import PROCESSED_PATH


def encode_midi(parameters: EncodeParameters) -> dict:
    """
    Encode a MIDI file into a symbolic representation with harmonic analysis.

    This function processes a MIDI file through several stages:
    1. MIDI Loading and Basic Processing:
       - Loads and quantizes MIDI data
       - Extracts timing information (beats and downbeats)
    2. Harmonic Analysis:
       - Extracts chord progressions
       - Analyzes tonal plan (key changes)
    3. REMI (REvamped MIDI) Encoding:
       - Combines notes, chords, and keys into a unified representation
       - Groups events by musical bars
       - Creates a sequence of musical events

    Args:
        parameters (EncodeParameters): Configuration parameters including:
            - midi (str or PrettyMIDI): MIDI file path or loaded MIDI object
            - alpha, beta, gamma (float): Key detection parameters
            - c, w (float): Tonal plan estimation parameters
            - save (bool): Whether to save the encoded output
            - encodings_out_dir (str): Directory for saving processed outputs

    Returns:
        dict: Status message indicating successful encoding
    """
    # Initialize encoder model
    encoder = MIDIEncoderModule(
        alpha=parameters.alpha,
        beta=parameters.beta,
        gamma=parameters.gamma,
        c=parameters.c,
        w=parameters.w,
        save=parameters.save,
        encodings_out_dir=parameters.encodings_out_dir,
    )

    # Process the MIDI file
    encoder.encode_midi(parameters.midi)

    return {"Message": "Midi Encoded Successfully"}


async def encode_dataset(config_path: str) -> dict:
    """
    Asynchronously encode an entire dataset of MIDI files with error handling and batch processing.

    This function processes a collection of MIDI files through several stages:
    1. Directory Setup:
       - Creates output directory for processed files
       - Sets up a dump directory for failed files
    2. Batch Processing:
       - Processes files in small batches to manage memory
       - Handles errors gracefully by moving failed files to dump directory
    3. Progress Tracking:
       - Maintains counts of processed and successful files
       - Collects error messages for failed files

    Args:
        parameters (EncodeDatasetParameters): Configuration parameters including:
            - config_path (str): Path to the configuration file

    Returns:
        dict: Processing summary including:
            - Total number of files processed
            - Number of successful encodings
            - Number of failed files
            - List of error messages
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    encoder_config = config.get("encoder", {})
    dataloader_config = config.get("dataloader", {})
    dataset_name = dataloader_config.get("dataset_name")

    # Set up directory paths
    processed_out_dir = os.path.join(PROCESSED_PATH, dataset_name)
    os.makedirs(processed_out_dir, exist_ok=True)

    # Initialize encoder model
    encoder = MIDIEncoderModule(
        alpha=encoder_config.get("alpha", 1.0),
        beta=encoder_config.get("beta", 1.0),
        gamma=encoder_config.get("gamma", 1.0),
        c=encoder_config.get("c", 1.0),
        w=encoder_config.get("w", 1.0),
        save=True,
        encodings_out_dir=processed_out_dir,
        device=encoder_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Initialize dataloader
    dataloader = DataloaderModule(
        dataset_name=dataset_name,
        context_size=dataloader_config.get("context_size", 512),
        max_positions=dataloader_config.get("max_positions", 512),
        max_bars=dataloader_config.get("max_bars", 512),
        max_bars_per_context=dataloader_config.get("max_bars_per_context", 16),
        max_contexts_per_file=dataloader_config.get("max_contexts_per_file", 16),
        bar_token_mask=dataloader_config.get("bar_token_mask", True),
        bar_token_idx=dataloader_config.get("bar_token_idx", 0),
        batch_size=encoder_config.get("batch_size", 10),
        num_workers=dataloader_config.get("num_workers", 4),
        pin_memory=dataloader_config.get("pin_memory", True),
        train_val_test_split=dataloader_config.get("train_val_test_split", (0.8, 0.1, 0.1)),
        load_latent=False,
        load_symb=False,
        load_emotions=False,
        encode=True,  # Enable encoding mode
    )

    # Set up trainer
    trainer = Trainer(
        accelerator=encoder_config.get("device", "gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        logger=False,
    )

    # Run prediction
    results = trainer.predict(encoder, datamodule=dataloader)

    # Process results
    total_files = 0
    successful = 0
    errors = []

    for batch_results in results:
        for result in batch_results:
            total_files += 1
            if result["success"]:
                successful += 1
            else:
                errors.append(f"Error processing {os.path.basename(result['file'])}: {result['error']}")

    return {"Message": "Dataset Encoded Successfully", "Total Files": total_files, "Successful": successful, "Failed": len(errors), "Errors": errors}
