from fastapi import Query
from typing import Annotated, Optional, Union
from pydantic import BaseModel, Field
import pretty_midi as pm

from domain.models.base_model import BaseEnum


class SymbolicFeaturesParameters(BaseModel):
    midi: str
    processed_dir: Annotated[
        Optional[str], Query(description="Directory containing the processed pkl file. If not provided, will use PROCESSED_PATH/dataset_name")
    ] = None
    save: Annotated[bool, Query(description="Whether to save the output")] = True
    level: Annotated[str, Query(description="Level of features to extract")] = "piece"
    add_position_tokens: Annotated[bool, Query(description="Whether to add position tokens before features (only applies to bar-level features)")] = False


class VaeTrainingParameters(BaseModel):
    """Parameters for training VAE models."""
    
    # Model architecture
    d_model: int = Field(512, description="Model dimension")
    context_size: int = Field(512, description="Context size")
    batch_size: int = Field(32, description="Batch size")
    num_attention_heads: int = Field(8, description="Number of attention heads")
    n_codes: int = Field(256, description="Number of codes")
    n_groups: int = Field(4, description="Number of groups")
    d_latent: int = Field(128, description="Latent dimension")
    encoder_layers: int = Field(6, description="Number of encoder layers")
    decoder_layers: int = Field(6, description="Number of decoder layers")
    encoder_ffn_dim: int = Field(2048, description="Encoder FFN dimension")
    decoder_ffn_dim: int = Field(2048, description="Decoder FFN dimension")
    dropout: float = Field(0.1, description="Dropout rate")
    
    # Training parameters
    lr: float = Field(1e-4, description="Learning rate")
    lr_schedule: str = Field("sqrt_decay", description="Learning rate schedule")
    warmup_steps: int = Field(4000, description="Warmup steps")
    max_steps: int = Field(100000, description="Maximum steps")
    max_epochs: int = Field(100, description="Maximum epochs")
    weight_decay: float = Field(1e-4, description="Weight decay")
    
    # VAE specific parameters
    windowed_attention_pr: float = Field(0.0, description="Windowed attention probability")
    max_lookahead: int = Field(4, description="Maximum lookahead")
    disable_vq: bool = Field(False, description="Disable vector quantization")
    beta: float = Field(0.02, description="Beta parameter for VQ-VAE")
    cycle_length: int = Field(2000, description="Cycle length")
    decay: float = Field(0.99, description="Decay rate")
    eps: float = Field(1e-5, description="Epsilon for numerical stability")
    restart_threshold: float = Field(1.0, description="Restart threshold")
    
    # Training configuration
    gpus: int = Field(1, description="Number of GPUs")
    accumulate_grad_batches: int = Field(1, description="Gradient accumulation batches")
    val_check_interval: int = Field(1000, description="Validation check interval")
    log_every_n_steps: int = Field(50, description="Log every n steps")
    save_top_k: int = Field(2, description="Save top k models")
    limit_val_batches: int = Field(64, description="Limit validation batches")
    num_sanity_val_steps: int = Field(2, description="Number of sanity validation steps")
    
    # Data parameters
    dataset_name: str = Field("ReMIDICaps", description="Dataset name")
    num_workers: int = Field(4, description="Number of workers")
    pin_memory: bool = Field(True, description="Pin memory")
    load_latent: bool = Field(False, description="Load latent representations")
    load_symb: bool = Field(False, description="Load symbolic features")
    
    # Device and performance
    device: str = Field("cuda", description="Device to use")
    
    # Checkpoint configuration
    checkpoint_dir: str = Field("output/checkpoints/vae", description="Checkpoint directory")
    load_from_checkpoint: bool = Field(False, description="Whether to load from checkpoint")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint file")
    weights_path: Optional[str] = Field(None, description="Path to model weights")
    config_path: Optional[str] = Field(None, description="Path to model configuration")


class LatentRepresentationParameters(BaseModel):
    """Parameters for generating latent representations."""
    
    # Model configuration
    checkpoint_path: str = Field(..., description="Path to trained VAE checkpoint")
    
    # Data parameters
    dataset_name: str = Field("ReMIDICaps", description="Dataset name")
    context_size: int = Field(-1, description="Context size (-1 for full context)")
    batch_size: int = Field(32, description="Batch size")
    num_workers: int = Field(4, description="Number of workers")
    pin_memory: bool = Field(True, description="Pin memory")
    
    # Processing parameters
    encode: bool = Field(False, description="Encode data")
    load_latent: bool = Field(False, description="Load latent representations")
    load_symb: bool = Field(False, description="Load symbolic features")
    
    # Device configuration
    device: str = Field("cuda", description="Device to use")
    
    # Output configuration
    output_dir: str = Field("output/latent_representations", description="Output directory")
    save_latents: bool = Field(True, description="Save latent representations")
    save_codes: bool = Field(True, description="Save VQ codes")
    
    # Processing configuration
    max_files: Optional[int] = Field(None, description="Maximum number of files to process")
    resume_from: Optional[str] = Field(None, description="Resume processing from specific file")


class SymbolicFeaturesDatasetParameters(BaseModel):
    """Parameters for extracting symbolic features from an entire dataset."""
    
    # Dataset configuration
    dataset_name: str = Field("ReMIDICaps", description="Name of the dataset")
    
    # Feature extraction configuration
    level: str = Field("bar", description="Level of features to extract (piece or bar)")
    add_position_tokens: bool = Field(False, description="Whether to add position tokens before features (only applies to bar-level features)")
    omit_time_sig: bool = Field(False, description="Whether to omit time signature features")
    omit_instruments: bool = Field(False, description="Whether to omit instrument features")
    omit_chords: bool = Field(False, description="Whether to omit chord features")
    omit_meta: bool = Field(False, description="Whether to omit meta features")
    
    # Processing configuration
    batch_size: int = Field(32, description="Batch size for processing")
    num_workers: int = Field(4, description="Number of workers for parallel processing")
    max_files: Optional[int] = Field(None, description="Maximum number of files to process")
    resume_from: Optional[str] = Field(None, description="Resume processing from specific file")
    
    # Input/Output configuration
    processed_dir: Optional[str] = Field(None, description="Directory containing processed pkl files")
    save: bool = Field(True, description="Whether to save the output")
    output_dir: Optional[str] = Field(None, description="Output directory for symbolic features")
    overwrite_existing: bool = Field(False, description="Overwrite existing feature files")
    
    # Device configuration
    device: str = Field("cuda", description="Device to use for processing")
    
    # Validation configuration
    validate_output: bool = Field(True, description="Validate extracted features")
    skip_invalid: bool = Field(True, description="Skip invalid files")