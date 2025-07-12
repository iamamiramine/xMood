"""
Base Parameter Classes for PA-AI-2 Pipeline

This module defines the hierarchical base classes for common parameters across
the pipeline services. Following the pipeline flow: dataloader → encoder → 
feature extraction → generator → multimodal mapping.

Each service inherits from the previous service's parameters and adds its own
specific parameters, ensuring consistency and single source of truth.
"""

from typing import Optional, Tuple
from pydantic import BaseModel, Field

# Import constants
from domain.constants.model_constants import ModelConstants
from domain.constants.encoder.midi_constants import MAX_N_BARS, DEFAULT_POS_PER_QUARTER


class BaseParameters(BaseModel):
    """Base parameters class with common configuration"""
    
    class Config:
        # Allow arbitrary types for enum compatibility
        arbitrary_types_allowed = True
        # Use enum values in JSON serialization
        use_enum_values = True


class DatasetParameters(BaseParameters):
    """Dataset-related parameters - foundation for all services"""
    
    dataset_name: str = Field("DATASET_NAME", description="Name of the dataset")
    device: str = Field("cuda", description="Device to use for processing")


class ModelArchitectureParameters(BaseParameters):
    """Model architecture parameters - shared across training models"""
    
    # Core architecture
    d_model: int = Field(ModelConstants.DEFAULT_D_MODEL, description="Model dimension")
    context_size: int = Field(ModelConstants.DEFAULT_CONTEXT_SIZE, description="Context size")
    max_positions: int = Field(ModelConstants.DEFAULT_MAX_POSITIONS, description="Maximum position embeddings")
    max_bars: int = Field(MAX_N_BARS, description="Maximum number of bars")
    
    # Attention and layers
    num_attention_heads: int = Field(ModelConstants.DEFAULT_NUM_ATTENTION_HEADS, description="Number of attention heads")
    encoder_layers: int = Field(ModelConstants.DEFAULT_ENCODER_LAYERS, description="Number of encoder layers")
    decoder_layers: int = Field(ModelConstants.DEFAULT_DECODER_LAYERS, description="Number of decoder layers")
    intermediate_size: int = Field(ModelConstants.DEFAULT_INTERMEDIATE_SIZE, description="Intermediate size")
    dropout: float = Field(ModelConstants.DEFAULT_DROPOUT, description="Dropout rate")
    
    # Vocabulary
    vocab_size: int = Field(ModelConstants.DEFAULT_VOCAB_SIZE, description="Vocabulary size")


class TrainingParameters(BaseParameters):
    """Training-related parameters - shared across training models"""
    
    # Basic training config
    batch_size: int = Field(ModelConstants.DEFAULT_BATCH_SIZE, description="Batch size")
    num_workers: int = Field(4, description="Number of workers")
    pin_memory: bool = Field(True, description="Pin memory")
    
    # Learning parameters
    lr: float = Field(ModelConstants.DEFAULT_LEARNING_RATE, description="Learning rate")
    lr_schedule: str = Field("sqrt_decay", description="Learning rate schedule")
    warmup_steps: int = Field(ModelConstants.DEFAULT_WARMUP_STEPS, description="Warmup steps")
    max_steps: int = Field(ModelConstants.DEFAULT_MAX_STEPS, description="Maximum steps")
    max_epochs: int = Field(100, description="Maximum epochs")
    weight_decay: float = Field(ModelConstants.DEFAULT_WEIGHT_DECAY, description="Weight decay")
    
    # GPU and parallelization
    gpus: int = Field(1, description="Number of GPUs")
    accumulate_grad_batches: int = Field(1, description="Gradient accumulation batches")


class DataProcessingParameters(BaseParameters):
    """Data processing parameters - shared across data handling services"""
    
    # Context and sequence configuration
    max_bars_per_context: int = Field(-1, description="Maximum bars per context")
    max_contexts_per_file: int = Field(-1, description="Maximum contexts per file")
    
    # Token configuration
    bar_token_mask: Optional[str] = Field(None, description="Bar masking token")
    bar_token_idx: int = Field(ModelConstants.BOS_TOKEN_ID, description="Bar token index")
    
    # Data split configuration
    train_val_test_split: Tuple[float, float, float] = Field(
        (0.7, 0.2, 0.1), 
        description="Train/validation/test split"
    )


class DataLoadingParameters(BaseParameters):
    """Data loading flags - shared across services that load data"""
    
    # Data loading configuration
    load_latent: bool = Field(True, description="Whether to load latent representations")
    load_symb: bool = Field(True, description="Whether to load symbolic features")
    load_emotions: bool = Field(True, description="Whether to load emotion data")
    load_global_features: bool = Field(False, description="Whether to load global features")
    load_text_prompts: bool = Field(False, description="Whether to load text prompts")
    
    # Processing modes
    encode: bool = Field(False, description="Whether to encode data")
    caption: bool = Field(False, description="Whether to generate captions")


class CheckpointParameters(BaseParameters):
    """Checkpoint and model loading parameters - shared across training models"""
    
    # Checkpoint configuration
    load_from_checkpoint: bool = Field(False, description="Whether to load from checkpoint")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint file")
    weights_path: Optional[str] = Field(None, description="Path to model weights")
    config_path: Optional[str] = Field(None, description="Path to model configuration")
    load_weights: bool = Field(False, description="Load weights instead of full checkpoint")


class ValidationParameters(BaseParameters):
    """Validation and logging parameters - shared across training models"""
    
    # Validation configuration
    val_check_interval: int = Field(ModelConstants.DEFAULT_VAL_CHECK_INTERVAL, description="Validation check interval")
    log_every_n_steps: int = Field(ModelConstants.DEFAULT_LOG_EVERY_N_STEPS, description="Log every n steps")
    save_top_k: int = Field(ModelConstants.DEFAULT_SAVE_TOP_K, description="Save top k models")
    limit_val_batches: int = Field(ModelConstants.DEFAULT_VALIDATION_BATCHES, description="Limit validation batches")
    num_sanity_val_steps: int = Field(ModelConstants.DEFAULT_SANITY_VAL_STEPS, description="Number of sanity validation steps")
    every_n_train_steps: int = Field(ModelConstants.DEFAULT_CHECKPOINT_EVERY_N_STEPS, description="Checkpoint every n train steps")


# Hierarchical base classes following the pipeline flow

class DataloaderBaseParameters(
    DatasetParameters,
    DataProcessingParameters,
    DataLoadingParameters,
    TrainingParameters
):
    """Base parameters for dataloader service - foundation of the pipeline"""
    
    # Override num_workers for dataloader (historically uses 2)
    num_workers: int = Field(2, description="Number of workers to load the dataset")


class EncoderBaseParameters(
    DataloaderBaseParameters,
    DatasetParameters
):
    """Base parameters for encoder service - inherits from dataloader"""
    
    # Override num_workers for encoder (uses 4)
    num_workers: int = Field(4, description="Number of workers for parallel processing")


class FeatureExtractionBaseParameters(
    EncoderBaseParameters,
    ModelArchitectureParameters,
    CheckpointParameters,
    ValidationParameters
):
    """Base parameters for feature extraction service - inherits from encoder, adds training"""
    
    # Specific defaults for feature extraction
    checkpoint_dir: str = Field("output/checkpoints/vae", description="Checkpoint directory")
    
    # Override data loading for feature extraction (historically uses False)
    load_latent: bool = Field(False, description="Whether to load latent representations")
    load_symb: bool = Field(False, description="Whether to load symbolic features")
    load_emotions: bool = Field(False, description="Whether to load emotion data")


class GeneratorBaseParameters(
    FeatureExtractionBaseParameters
):
    """Base parameters for generator service - inherits from feature extraction"""
    
    # Override specific parameters for generator
    max_bars: int = Field(16, description="Maximum bars")  # Generator historically uses 16
    checkpoint_dir: str = Field("output/checkpoints/generator", description="Checkpoint directory")
    
    # Override data loading for generator (back to True)
    load_latent: bool = Field(True, description="Whether to load latent representations")
    load_symb: bool = Field(True, description="Whether to load symbolic features")
    load_emotions: bool = Field(True, description="Whether to load emotion data")
    
    # Override checkpoint every n steps (generator uses 1000)
    every_n_train_steps: int = Field(1000, description="Checkpoint every n train steps")


class MultimodalBaseParameters(
    GeneratorBaseParameters
):
    """Base parameters for multimodal service - inherits from generator"""
    
    # Override specific parameters for multimodal
    checkpoint_dir: str = Field("output/checkpoints/multimodal_mapping", description="Checkpoint directory")
    lr_schedule: str = Field("cosine", description="Learning rate schedule")
    
    # Override data loading for multimodal (adds global features and text prompts)
    load_global_features: bool = Field(True, description="Whether to load global features")
    load_text_prompts: bool = Field(True, description="Whether to load text prompts")
    
    # Additional multimodal-specific parameters
    image_dim: int = Field(ModelConstants.DEFAULT_IMAGE_DIM, description="Image feature dimension")
    text_dim: int = Field(ModelConstants.DEFAULT_TEXT_DIM, description="Text feature dimension")
    fusion_dim: int = Field(ModelConstants.DEFAULT_FUSION_DIM, description="Fusion dimension")
    latent_dim: int = Field(ModelConstants.DEFAULT_LATENT_DIM, description="Latent dimension")
    output_dim: int = Field(ModelConstants.DEFAULT_OUTPUT_DIM, description="Output dimension")
    
    # Multimodal-specific dimensions
    global_feature_dim: int = Field(ModelConstants.DEFAULT_GLOBAL_FEATURE_DIM, description="Global feature dimension")
    global_feature_out_dim: int = Field(ModelConstants.DEFAULT_GLOBAL_FEATURE_OUT_DIM, description="Global feature output dimension")
    mood_dim: int = Field(ModelConstants.DEFAULT_MOOD_DIM, description="Mood feature dimension")
    mood_out_dim: int = Field(ModelConstants.DEFAULT_MOOD_OUT_DIM, description="Mood feature output dimension")
    
    # Multimodal-specific training parameters
    fusion_heads: int = Field(ModelConstants.DEFAULT_NUM_ATTENTION_HEADS, description="Number of fusion heads")
    fusion_type: str = Field("linear_concat", description="Fusion type")
    num_layers: int = Field(4, description="Number of layers")
    
    # Modality dropout rates
    image_dropout_rate: float = Field(0.2, description="Image dropout rate")
    text_dropout_rate: float = Field(0.2, description="Text dropout rate")
    global_feature_dropout_rate: float = Field(0.2, description="Global feature dropout rate")
    mood_dropout_rate: float = Field(0.2, description="Mood dropout rate")
    
    # KL annealing parameters
    use_kl_annealing: bool = Field(True, description="Use KL annealing")
    kl_start: float = Field(0.0, description="KL annealing start value")
    kl_end: float = Field(1.0, description="KL annealing end value")
    kl_anneal_steps: int = Field(10000, description="KL annealing steps") 