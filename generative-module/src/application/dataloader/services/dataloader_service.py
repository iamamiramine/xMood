import os
import traceback
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging

from application.dataloader.models.dataloader_model import DataloaderModule
from application.dataloader.helper.dataloader_helper import represent_encoding
from domain.models.dataloader.dataloader_model import (
    DataloaderModuleParameters,
    DatasetLoadParameters,
    DataloaderParameters
)
from domain.constants.paths_constants import PROCESSED_PATH, LABELS_PATH, MIDI_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_dataloader_module(
    parameters: Union[DataloaderModuleParameters, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Initialize a DataloaderModule with the specified parameters.
    
    Args:
        parameters: DataloaderModuleParameters or dict with dataloader configuration
        
    Returns:
        Dictionary with initialization status and module information
    """
    try:
        # Convert parameters to correct type if needed
        if isinstance(parameters, dict):
            params = DataloaderModuleParameters(**parameters)
        else:
            params = parameters
        
        # Initialize the dataloader module
        dataloader_module = DataloaderModule(
            dataset_name=params.dataset_name,
            context_size=params.context_size,
            max_positions=params.max_positions,
            max_bars=params.max_bars,
            max_bars_per_context=params.max_bars_per_context,
            max_contexts_per_file=params.max_contexts_per_file,
            bar_token_mask=params.bar_token_mask,
            bar_token_idx=params.bar_token_idx,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
            train_val_test_split=params.train_val_test_split,
            load_latent=params.load_latent,
            load_symb=params.load_symb,
            load_emotions=params.load_emotions,
            load_global_features=params.load_global_features,
            load_text_prompts=params.load_text_prompts,
            encode=params.encode,
            caption=params.caption,
        )
        
        # Setup the dataloader
        dataloader_module.setup()
        
        # Get dataset information
        dataset_info = {
            "dataset_name": params.dataset_name,
            "total_files": len(dataloader_module.midi_files),
            "train_files": len(dataloader_module.train_ds.datapipe.files) if hasattr(dataloader_module.train_ds.datapipe, 'files') else 0,
            "valid_files": len(dataloader_module.valid_ds.datapipe.files) if hasattr(dataloader_module.valid_ds.datapipe, 'files') else 0,
            "test_files": len(dataloader_module.test_ds.datapipe.files) if hasattr(dataloader_module.test_ds.datapipe, 'files') else 0,
            "context_size": params.context_size,
            "batch_size": params.batch_size,
            "load_latent": params.load_latent,
            "load_symb": params.load_symb,
            "load_emotions": params.load_emotions,
            "load_global_features": params.load_global_features,
            "load_text_prompts": params.load_text_prompts,
        }
        
        logger.info(f"DataloaderModule initialized successfully for dataset: {params.dataset_name}")
        
        return {
            "status": "success",
            "message": f"DataloaderModule initialized successfully for dataset: {params.dataset_name}",
            "dataset_info": dataset_info,
            "dataloader_module": dataloader_module  # Return the actual module for use
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize DataloaderModule: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to initialize DataloaderModule: {str(e)}",
            "error": str(e)
        }


def load_dataset(
    parameters: Union[DatasetLoadParameters, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Load and process a dataset with the specified parameters.
    
    Args:
        parameters: DatasetLoadParameters or dict with dataset loading configuration
        
    Returns:
        Dictionary with loading status and dataset information
    """
    try:
        # Convert parameters to correct type if needed
        if isinstance(parameters, dict):
            params = DatasetLoadParameters(**parameters)
        else:
            params = parameters
        
        # First initialize the dataloader module
        dataloader_params = DataloaderModuleParameters(
            dataset_name=params.dataset_name,
            load_latent=params.load_latent,
            load_symb=params.load_symb,
            load_emotions=params.load_emotions,
            load_global_features=params.load_global_features,
            load_text_prompts=params.load_text_prompts,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
        )
        
        # Initialize the dataloader
        init_result = initialize_dataloader_module(dataloader_params)
        if init_result["status"] != "success":
            return init_result
        
        dataloader_module = init_result["dataloader_module"]
        
        # Get the appropriate dataloader based on stage
        if params.stage == "train":
            dataloader = dataloader_module.train_dataloader()
        elif params.stage == "val":
            dataloader = dataloader_module.val_dataloader()
        elif params.stage == "test":
            dataloader = dataloader_module.test_dataloader()
        elif params.stage == "predict":
            dataloader = dataloader_module.predict_dataloader()
        else:
            dataloader = dataloader_module.train_dataloader()
        
        # Process batches if output directory is specified
        processed_batches = 0
        if params.output_dir and params.save_processed:
            os.makedirs(params.output_dir, exist_ok=True)
            
            for batch_idx, batch in enumerate(dataloader):
                if params.save_processed:
                    batch_file = os.path.join(params.output_dir, f"batch_{batch_idx}.pt")
                    import torch
                    torch.save(batch, batch_file)
                processed_batches += 1
        
        logger.info(f"Dataset loaded successfully: {params.dataset_name}")
        
        return {
            "status": "success",
            "message": f"Dataset loaded successfully: {params.dataset_name}",
            "dataset_info": init_result["dataset_info"],
            "stage": params.stage,
            "processed_batches": processed_batches,
            "dataloader": dataloader  # Return the actual dataloader for use
        }
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to load dataset: {str(e)}",
            "error": str(e)
        }


def process_dataset_files(
    parameters: Union[DataloaderParameters, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Process individual files in a dataset using the dataloader helper functions.
    
    Args:
        parameters: DataloaderParameters or dict with processing configuration
        
    Returns:
        Dictionary with processing status and results
    """
    try:
        # Convert parameters to correct type if needed
        if isinstance(parameters, dict):
            params = DataloaderParameters(**parameters)
        else:
            params = parameters
        
        # Get dataset files
        import glob
        import pandas as pd
        
        # Path to the CSV file
        csv_file_path = os.path.join(LABELS_PATH, f"{params.dataset_name}.csv")
        
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        # Read CSV file
        labels_df = pd.read_csv(csv_file_path)
        csv_file_names = set(labels_df["file"].values)
        
        # Find MIDI files
        midi_files = []
        for ext in ["*.mid", "*.midi"]:
            for f in glob.glob(os.path.join(MIDI_PATH, f"**/{ext}"), recursive=True):
                if os.path.basename(f) in csv_file_names:
                    midi_files.append(f)
        
        processed_files = 0
        failed_files = []
        
        # Process each file
        for midi_file in midi_files:
            try:
                # Load processed data
                processed_file = os.path.join(
                    str(PROCESSED_PATH),
                    params.dataset_name,
                    f"{os.path.basename(midi_file)}_processed.pkl"
                )
                
                if os.path.exists(processed_file):
                    import pickle
                    processed_data = pickle.load(open(processed_file, "rb"))
                    
                    if processed_data and "encodings" in processed_data:
                        # Process the encoding
                        encoding = processed_data["encodings"]
                        events = encoding["events"]
                        
                        # Use represent_encoding helper function
                        result = represent_encoding(
                            midi=os.path.basename(midi_file),
                            dataset_name=params.dataset_name,
                            context_size=params.context_size,
                            max_bars=params.max_bars,
                            max_positions=params.max_positions,
                            bar_token_mask=params.bar_token_mask,
                            max_bars_per_context=params.max_bars_per_context,
                            max_contexts_per_file=params.max_contexts_per_file,
                            events=events,
                            save=False,
                            api_call=True
                        )
                        
                        processed_files += 1
                
            except Exception as e:
                logger.warning(f"Failed to process file {midi_file}: {str(e)}")
                failed_files.append(midi_file)
                continue
        
        logger.info(f"Processed {processed_files} files from dataset: {params.dataset_name}")
        
        return {
            "status": "success",
            "message": f"Processed {processed_files} files from dataset: {params.dataset_name}",
            "processed_files": processed_files,
            "failed_files": len(failed_files),
            "total_files": len(midi_files),
            "dataset_name": params.dataset_name
        }
        
    except Exception as e:
        logger.error(f"Failed to process dataset files: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to process dataset files: {str(e)}",
            "error": str(e)
        } 