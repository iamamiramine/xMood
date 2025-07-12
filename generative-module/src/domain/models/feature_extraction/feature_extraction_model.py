from fastapi import Query
from typing import Annotated, Optional, Union
from pydantic import BaseModel, Field
import pretty_midi as pm

from domain.models.base_model import BaseEnum

# Import constants
from domain.constants.model_constants import ModelConstants
from domain.constants.encoder.midi_constants import MAX_N_BARS, DEFAULT_POS_PER_QUARTER

# Import base parameter classes
from domain.models.base_parameters import FeatureExtractionBaseParameters


class SymbolicFeaturesParameters(BaseModel):
    """Parameters for single file symbolic feature extraction - lightweight API model"""
    
    midi: str
    processed_dir: Annotated[
        Optional[str], Query(description="Directory containing the processed pkl file. If not provided, will use PROCESSED_PATH/dataset_name")
    ] = None
    save: Annotated[bool, Query(description="Whether to save the output")] = True
    level: Annotated[str, Query(description="Level of features to extract")] = "piece"
    add_position_tokens: Annotated[bool, Query(description="Whether to add position tokens before features (only applies to bar-level features)")] = False


class VaeTrainingParameters(FeatureExtractionBaseParameters):
    """Parameters for training VAE models - inherits from base"""
    
    # VAE-specific architecture parameters
    n_codes: int = Field(ModelConstants.DEFAULT_N_CODES, description="Number of codes")
    n_groups: int = Field(ModelConstants.DEFAULT_N_GROUPS, description="Number of groups")
    d_latent: int = Field(ModelConstants.DEFAULT_LATENT_DIM, description="Latent dimension")
    encoder_ffn_dim: int = Field(ModelConstants.DEFAULT_INTERMEDIATE_SIZE, description="Encoder FFN dimension")
    decoder_ffn_dim: int = Field(ModelConstants.DEFAULT_INTERMEDIATE_SIZE, description="Decoder FFN dimension")
    
    # VAE specific parameters
    windowed_attention_pr: float = Field(0.0, description="Windowed attention probability")
    max_lookahead: int = Field(4, description="Maximum lookahead")
    disable_vq: bool = Field(False, description="Disable vector quantization")
    beta: float = Field(0.02, description="Beta parameter for VQ-VAE")
    cycle_length: int = Field(2000, description="Cycle length")
    decay: float = Field(ModelConstants.DEFAULT_VQ_DECAY, description="Decay rate")
    eps: float = Field(ModelConstants.DEFAULT_VQ_EPS, description="Epsilon for numerical stability")
    restart_threshold: float = Field(ModelConstants.DEFAULT_VQ_RESTART_THRESHOLD, description="Restart threshold")
    position_embedding_type: str = Field("relative_key_query", description="Position embedding type")
    automatic_optimization: bool = Field(False, description="Use automatic optimization")
    
    # Additional logging configuration
    use_wandb: bool = Field(False, description="Use Weights & Biases logging")


class LatentRepresentationParameters(FeatureExtractionBaseParameters):
    """Parameters for generating latent representations - inherits from base"""
    
    # Model configuration
    checkpoint_path: str = Field(..., description="Path to trained VAE checkpoint")
    
    # Override context size for latent generation (can be -1 for full context)
    context_size: int = Field(-1, description="Context size (-1 for full context)")
    
    # Output configuration
    output_dir: str = Field("output/latent_representations", description="Output directory")
    save_latents: bool = Field(True, description="Save latent representations")
    save_codes: bool = Field(True, description="Save VQ codes")
    
    # Processing configuration
    max_files: Optional[int] = Field(None, description="Maximum number of files to process")
    resume_from: Optional[str] = Field(None, description="Resume processing from specific file")
    
    # Training configuration for prediction
    max_steps: int = Field(1000, description="Maximum steps for prediction")
    val_check_interval: int = Field(100, description="Validation check interval")
    log_every_n_steps: int = Field(10, description="Log every n steps")
    limit_val_batches: int = Field(ModelConstants.DEFAULT_VALIDATION_BATCHES, description="Limit validation batches")
    num_sanity_val_steps: int = Field(ModelConstants.DEFAULT_SANITY_VAL_STEPS, description="Number of sanity validation steps")
    save_top_k: int = Field(2, description="Save top k models")
    every_n_train_steps: int = Field(100, description="Checkpoint every n train steps")


class SymbolicFeaturesDatasetParameters(FeatureExtractionBaseParameters):
    """Parameters for extracting symbolic features from an entire dataset - inherits from base"""
    
    # Feature extraction configuration
    level: str = Field("bar", description="Level of features to extract (piece or bar)")
    add_position_tokens: bool = Field(False, description="Whether to add position tokens before features (only applies to bar-level features)")
    omit_time_sig: bool = Field(False, description="Whether to omit time signature features")
    omit_instruments: bool = Field(False, description="Whether to omit instrument features")
    omit_chords: bool = Field(False, description="Whether to omit chord features")
    omit_meta: bool = Field(False, description="Whether to omit meta features")
    
    # Processing configuration
    max_files: Optional[int] = Field(None, description="Maximum number of files to process")
    resume_from: Optional[str] = Field(None, description="Resume processing from specific file")
    
    # Input/Output configuration
    processed_dir: Optional[str] = Field(None, description="Directory containing processed pkl files")
    save: bool = Field(True, description="Whether to save the output")
    output_dir: Optional[str] = Field(None, description="Output directory for symbolic features")
    overwrite_existing: bool = Field(False, description="Overwrite existing feature files")
    
    # Validation configuration
    validate_output: bool = Field(True, description="Validate extracted features")
    skip_invalid: bool = Field(True, description="Skip invalid files")