from typing import Optional
from pydantic import BaseModel, Field

# Import constants
from domain.constants.model_constants import ModelConstants
from domain.constants.midi_constants import MAX_N_BARS


class MultimodalMappingParameters(BaseModel):
    """
    Parameters for multimodal mapping service.
    
    Enhanced to handle the data formats from ReMIDICaps dataset.
    """
    # Output configuration
    output_folder: Optional[str] = Field(None, description="Folder to save generated features")
    output_name: str = Field("multimodal_features", description="Name of the output file")
    
    # Model configuration
    weights_path: str = Field(..., description="Path to the model weights")
    config_path: str = Field(..., description="Path to the model configuration file")
    
    # Generation parameters
    num_bars: int = Field(8, description="Number of bars to generate")
    temperature: float = Field(ModelConstants.DEFAULT_TEMPERATURE, description="Temperature for sampling during generation")


class MultimodalTrainingParameters(BaseModel):
    """
    Parameters for training multimodal mapping models.
    """
    # Model checkpoint configuration
    load_from_checkpoint: bool = Field(False, description="Whether to load from checkpoint")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint file")
    weights_path: Optional[str] = Field(None, description="Path to model weights")
    config_path: Optional[str] = Field(None, description="Path to model configuration")
    
    # Input dimensions
    image_dim: int = Field(ModelConstants.DEFAULT_IMAGE_DIM, description="Image feature dimension")
    text_dim: int = Field(ModelConstants.DEFAULT_TEXT_DIM, description="Text feature dimension")
    global_feature_dim: int = Field(ModelConstants.DEFAULT_GLOBAL_FEATURE_DIM, description="Global feature dimension")
    global_feature_out_dim: int = Field(ModelConstants.DEFAULT_GLOBAL_FEATURE_OUT_DIM, description="Global feature output dimension")
    mood_dim: int = Field(ModelConstants.DEFAULT_MOOD_DIM, description="Mood feature dimension")
    mood_out_dim: int = Field(ModelConstants.DEFAULT_MOOD_OUT_DIM, description="Mood feature output dimension")
    
    # Architecture parameters
    fusion_dim: int = Field(ModelConstants.DEFAULT_FUSION_DIM, description="Fusion dimension")
    d_model: int = Field(ModelConstants.DEFAULT_D_MODEL, description="Model dimension")
    latent_dim: int = Field(ModelConstants.DEFAULT_LATENT_DIM, description="Latent dimension")
    output_dim: int = Field(ModelConstants.DEFAULT_OUTPUT_DIM, description="Output dimension")
    context_size: int = Field(ModelConstants.DEFAULT_CONTEXT_SIZE, description="Context size")
    num_layers: int = Field(4, description="Number of layers")
    num_heads: int = Field(ModelConstants.DEFAULT_NUM_ATTENTION_HEADS, description="Number of attention heads")
    fusion_heads: int = Field(ModelConstants.DEFAULT_NUM_ATTENTION_HEADS, description="Number of fusion heads")
    fusion_type: str = Field("linear_concat", description="Fusion type")
    dropout: float = Field(ModelConstants.DEFAULT_DROPOUT, description="Dropout rate")
    training: bool = Field(True, description="Training mode")
    max_positions: int = Field(ModelConstants.DEFAULT_MAX_POSITIONS, description="Maximum position embeddings")
    max_bars: int = Field(MAX_N_BARS, description="Maximum number of bars")
    
    # Modality dropout rates
    image_dropout_rate: float = Field(0.2, description="Image dropout rate")
    text_dropout_rate: float = Field(0.2, description="Text dropout rate")
    global_feature_dropout_rate: float = Field(0.2, description="Global feature dropout rate")
    mood_dropout_rate: float = Field(0.2, description="Mood dropout rate")
    
    # Learning rate and schedule
    lr: float = Field(ModelConstants.DEFAULT_LEARNING_RATE, description="Learning rate")
    lr_schedule: str = Field("cosine", description="Learning rate schedule")
    warmup_steps: int = Field(ModelConstants.DEFAULT_WARMUP_STEPS, description="Warmup steps")
    max_steps: int = Field(ModelConstants.DEFAULT_MAX_STEPS, description="Maximum steps")
    
    # KL annealing
    use_kl_annealing: bool = Field(True, description="Use KL annealing")
    kl_start: float = Field(0.0, description="KL annealing start value")
    kl_end: float = Field(1.0, description="KL annealing end value")
    kl_anneal_steps: int = Field(10000, description="KL annealing steps")
    
    # Training parameters
    batch_size: int = Field(ModelConstants.DEFAULT_BATCH_SIZE, description="Batch size")
    max_epochs: int = Field(100, description="Maximum epochs")
    gpus: int = Field(1, description="Number of GPUs")
    accumulate_grad_batches: int = Field(1, description="Gradient accumulation batches")
    
    # Data parameters
    dataset_name: str = Field("DATASET_NAME", description="Dataset name")
    bar_token_idx: int = Field(ModelConstants.BOS_TOKEN_ID, description="Bar token index")
    train_val_test_split: tuple = Field((0.7, 0.2, 0.1), description="Train/validation/test split")
    
    # Additional data loading flags
    max_bars_per_context: int = Field(-1, description="Maximum bars per context")
    max_contexts_per_file: int = Field(-1, description="Maximum contexts per file")
    bar_token_mask: Optional[str] = Field(None, description="Bar masking token")
    
    # Device and performance
    device: str = Field("cuda", description="Device to use")
    num_workers: int = Field(4, description="Number of workers")
    pin_memory: bool = Field(True, description="Pin memory")
    
    # Checkpoint and logging
    checkpoint_dir: str = Field("output/checkpoints/multimodal", description="Checkpoint directory")
    log_every_n_steps: int = Field(ModelConstants.DEFAULT_LOG_EVERY_N_STEPS, description="Log every n steps")
    val_check_interval: int = Field(ModelConstants.DEFAULT_VAL_CHECK_INTERVAL, description="Validation check interval")
    save_top_k: int = Field(ModelConstants.DEFAULT_SAVE_TOP_K, description="Save top k models")
    every_n_train_steps: int = Field(1000, description="Checkpoint every n train steps")
    
    # Validation parameters
    limit_val_batches: int = Field(ModelConstants.DEFAULT_VALIDATION_BATCHES, description="Limit validation batches")
    num_sanity_val_steps: int = Field(ModelConstants.DEFAULT_SANITY_VAL_STEPS, description="Number of sanity validation steps") 