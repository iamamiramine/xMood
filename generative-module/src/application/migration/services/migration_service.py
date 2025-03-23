import os
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Union, List

import torch
from transformers import PreTrainedTokenizerFast

from application.migration.models.migration_model import (
    TokenizerParameters,
    DatasetParameters,
    ModelExportParameters,
    MigrationConfig,
    ProjectionModelParameters
)
from application.migration.helpers.tokenizer_helper import (
    create_and_save_tokenizer,
    convert_to_hf_tokenizer
)
from application.migration.helpers.dataset_helper import (
    prepare_dataset
)
from application.migration.helpers.model_helper import (
    export_model,
    create_extension_files
)
from domain.constants.encoder.token_constants import (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN
)


def migrate_tokenizer(params: TokenizerParameters) -> PreTrainedTokenizerFast:
    """Migrate the custom vocabularies to a HuggingFace tokenizer"""
    print(f"Creating tokenizer with vocabulary size {params.vocab_size}...")
    
    # Create the tokenizer directory if it doesn't exist
    os.makedirs(params.output_dir, exist_ok=True)
    
    # If using existing tokenizer, load it
    if params.use_existing and params.existing_tokenizer_path:
        print(f"Loading existing tokenizer from {params.existing_tokenizer_path}...")
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(params.existing_tokenizer_path)
            return tokenizer
        except Exception as e:
            print(f"Error loading existing tokenizer: {str(e)}")
            print("Falling back to creating a new tokenizer...")
    
    # Create a new tokenizer
    tokenizer_dir = create_and_save_tokenizer(params)
    print(f"Tokenizer created and saved to {tokenizer_dir}")
    
    # Load the HF tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    return tokenizer


def migrate_dataset(params: DatasetParameters, tokenizer: Optional[PreTrainedTokenizerFast] = None) -> Dict:
    """Prepare the dataset for training"""
    print(f"Creating dataset from {params.midi_dir} to {params.output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(params.output_dir, exist_ok=True)
    
    # Prepare the dataset
    dataset_info = prepare_dataset(params, tokenizer)
    
    print(f"Dataset created with {len(dataset_info['text']['train'])} training samples")
    if 'tokenized' in dataset_info:
        print(f"Tokenized dataset saved to {params.output_dir}")
    
    return dataset_info


def migrate_model(params: ModelExportParameters, tokenizer: PreTrainedTokenizerFast) -> Path:
    """Export the model for text-generation-webui"""
    print(f"Exporting model of type {params.model_type} to {params.output_dir}...")
    
    # Export the model
    model_dir = export_model(params, tokenizer)
    
    if params.webui_dir:
        print(f"Model exported to text-generation-webui at {params.webui_dir}/models/{params.model_name}")
        
        if params.convert_format:
            extension_dir = os.path.join(params.webui_dir, "extensions", "music_generator")
            print(f"Created extension at {extension_dir}")
    
    return model_dir


def run_migration(config_path: Union[str, Path]) -> Dict:
    """Run the full migration process using a config file"""
    print(f"Starting migration process using config {config_path}...")
    
    # Load config
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    # Parse config into our dataclasses
    tokenizer_params = TokenizerParameters(**config_data.get("tokenizer", {}))
    
    dataset_params = DatasetParameters(
        midi_dir=Path(config_data.get("dataset", {}).get("midi_dir", "")),
        output_dir=Path(config_data.get("dataset", {}).get("output_dir", "")),
        processed_dir=Path(config_data.get("dataset", {}).get("processed_dir", "")),
        **{k: v for k, v in config_data.get("dataset", {}).items() 
           if k not in ["midi_dir", "output_dir", "processed_dir"]}
    )
    
    model_export_params = ModelExportParameters(
        tokenizer_path=Path(config_data.get("model_export", {}).get("tokenizer_path", "")),
        **{k: v for k, v in config_data.get("model_export", {}).items() 
           if k != "tokenizer_path"}
    )
    
    # Add projection parameters if using integrated pipeline
    projection_params = None
    if config_data.get("migration", {}).get("integrated_pipeline", False) and "projection" in config_data:
        projection_params = ProjectionModelParameters(
            **config_data.get("projection", {})
        )
    
    migration_config = MigrationConfig(
        tokenizer=tokenizer_params,
        dataset=dataset_params,
        model_export=model_export_params,
        projection=projection_params,
        **config_data.get("migration", {})
    )
    
    # Step 1: Create tokenizer
    tokenizer = migrate_tokenizer(migration_config.tokenizer)
    
    # Step 2: Prepare dataset
    dataset_info = migrate_dataset(migration_config.dataset, tokenizer)
    
    # Step 3: Export model
    # For integrated pipeline, we need special handling
    if migration_config.integrated_pipeline:
        print("Using integrated pipeline approach...")
        # Handle projection model first
        if migration_config.projection:
            print(f"Processing projection model...")
            # Additional projection model setup could go here
    
    model_dir = migrate_model(migration_config.model_export, tokenizer)
    
    # Return summary of migration
    return {
        "tokenizer": str(tokenizer),
        "dataset": dataset_info,
        "model_dir": str(model_dir),
        "status": "Migration completed successfully",
        "integrated_pipeline": migration_config.integrated_pipeline
    }


def create_migration_config(
    output_dir: str,
    webui_dir: str,
    checkpoint_path: Optional[str] = None,
    projection_checkpoint: Optional[str] = None,
    generator_checkpoint: Optional[str] = None,
    model_name: str = "music_generation_model",
    model_type: str = "t5",
    create_tokenizer: bool = True,
    create_dataset: bool = True,
    midi_dir: Optional[str] = None,
    processed_dir: Optional[str] = None,
    integrated_pipeline: bool = False
) -> str:
    """Create a default migration configuration file"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create configuration
    config = {
        "tokenizer": {
            "vocab_size": 50000,
            "special_tokens": [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN],
            "output_dir": os.path.join(output_dir, "tokenizer"),
            "tokenizer_filename": "music_tokenizer.json",
            "use_bpe": True,
            "use_existing": False
        },
        
        "dataset": {
            "midi_dir": midi_dir or "",
            "processed_dir": processed_dir or "",
            "output_dir": os.path.join(output_dir, "dataset"),
            "split_ratio": [0.8, 0.1, 0.1],
            "max_samples": None,
            "include_metadata": True,
            "add_moods": True,
            "add_features": True,
            "include_chord_progression": True,
            "include_key_signature": True,
            "tokenize": True,
            "max_length": 1024,
            "create_projection_data": integrated_pipeline,
            "create_generation_data": True,
            "features_output_dir": os.path.join(output_dir, "features")
        },
        
        "model_export": {
            "model_type": model_type,
            "model_name": model_name,
            "source_checkpoint": checkpoint_path,
            "projection_checkpoint": projection_checkpoint,
            "generator_checkpoint": generator_checkpoint,
            "tokenizer_path": os.path.join(output_dir, "tokenizer", "hf_tokenizer"),
            "d_model": 512,
            "num_encoder_layers": 6,
            "num_decoder_layers": 12,
            "num_heads": 8,
            "d_ff": 2048,
            "output_dir": os.path.join(output_dir, "model"),
            "webui_dir": webui_dir,
            "convert_format": True,
            "use_integrated_pipeline": integrated_pipeline
        },
        
        "migration": {
            "create_extension": True,
            "extension_name": "music_generator",
            "convert_existing_checkpoints": checkpoint_path is not None or generator_checkpoint is not None,
            "existing_checkpoint_path": checkpoint_path,
            "integrated_pipeline": integrated_pipeline
        }
    }
    
    # Add projection model config if using integrated pipeline
    if integrated_pipeline:
        config["projection"] = {
            "d_model": 512,
            "encoder_layers": 6,
            "decoder_layers": 6,
            "num_heads": 8,
            "d_ff": 2048,
            "checkpoint_path": projection_checkpoint,
            "mood_vocab_size": 100,
            "feature_embedding_size": 1024,
            "context_size": 256,
            "max_bars": 512,
            "max_positions": 1024,
            "output_dir": os.path.join(output_dir, "projection_model")
        }
    
    # Save configuration
    config_path = os.path.join(output_dir, "migration_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Migration configuration saved to {config_path}")
    return config_path


def migrate_to_webui(
    output_dir: str,
    webui_dir: str,
    checkpoint_path: Optional[str] = None,
    projection_checkpoint: Optional[str] = None,
    generator_checkpoint: Optional[str] = None,
    model_name: str = "music_generation_model",
    model_type: str = "t5",
    create_config: bool = True,
    config_path: Optional[str] = None,
    integrated_pipeline: bool = False
) -> Dict:
    """
    Migrate music generation model to text-generation-webui
    
    Args:
        output_dir: Directory to store intermediate files
        webui_dir: Directory where text-generation-webui is installed
        checkpoint_path: Path to existing checkpoint (optional)
        projection_checkpoint: Path to projection model checkpoint (optional)
        generator_checkpoint: Path to generator model checkpoint (optional)
        model_name: Name for the exported model
        model_type: Type of model to create (t5, bart, custom)
        create_config: Whether to create a new config file
        config_path: Path to existing config file (if create_config=False)
        integrated_pipeline: Whether to use the integrated pipeline approach
    
    Returns:
        Dictionary with migration results
    """
    if create_config:
        config_path = create_migration_config(
            output_dir=output_dir,
            webui_dir=webui_dir,
            checkpoint_path=checkpoint_path,
            projection_checkpoint=projection_checkpoint,
            generator_checkpoint=generator_checkpoint,
            model_name=model_name,
            model_type=model_type,
            integrated_pipeline=integrated_pipeline
        )
    elif not config_path:
        raise ValueError("Either create_config must be True or config_path must be provided")
    
    # Run migration
    return run_migration(config_path) 