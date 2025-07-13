from typing import Optional
from pydantic import BaseModel, Field

# Import constants
from domain.constants.model_constants import ModelConstants
from domain.constants.midi_constants import MAX_N_BARS


class VaeTrainingParameters(BaseModel):
    """Parameters for training VAE models."""

    # Model architecture
    d_model: int = Field(ModelConstants.DEFAULT_D_MODEL, description="Model dimension")
    context_size: int = Field(ModelConstants.DEFAULT_CONTEXT_SIZE, description="Context size")
    batch_size: int = Field(ModelConstants.DEFAULT_BATCH_SIZE, description="Batch size")
    num_attention_heads: int = Field(
        ModelConstants.DEFAULT_NUM_ATTENTION_HEADS, description="Number of attention heads"
    )
    n_codes: int = Field(ModelConstants.DEFAULT_N_CODES, description="Number of codes")
    n_groups: int = Field(ModelConstants.DEFAULT_N_GROUPS, description="Number of groups")
    d_latent: int = Field(ModelConstants.DEFAULT_LATENT_DIM, description="Latent dimension")
    encoder_layers: int = Field(ModelConstants.DEFAULT_ENCODER_LAYERS, description="Number of symbolic layers")
    decoder_layers: int = Field(ModelConstants.DEFAULT_DECODER_LAYERS, description="Number of decoder layers")
    encoder_ffn_dim: int = Field(ModelConstants.DEFAULT_INTERMEDIATE_SIZE, description="Encoder FFN dimension")
    decoder_ffn_dim: int = Field(ModelConstants.DEFAULT_INTERMEDIATE_SIZE, description="Decoder FFN dimension")
    dropout: float = Field(ModelConstants.DEFAULT_DROPOUT, description="Dropout rate")
    max_positions: int = Field(ModelConstants.DEFAULT_MAX_POSITIONS, description="Maximum position embeddings")
    max_bars: int = Field(MAX_N_BARS, description="Maximum number of bars")

    # Training parameters
    lr: float = Field(ModelConstants.DEFAULT_LEARNING_RATE, description="Learning rate")
    lr_schedule: str = Field("sqrt_decay", description="Learning rate schedule")
    warmup_steps: int = Field(ModelConstants.DEFAULT_WARMUP_STEPS, description="Warmup steps")
    max_steps: int = Field(ModelConstants.DEFAULT_MAX_STEPS, description="Maximum steps")
    max_epochs: int = Field(100, description="Maximum epochs")
    weight_decay: float = Field(ModelConstants.DEFAULT_WEIGHT_DECAY, description="Weight decay")

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

    # Training configuration
    gpus: int = Field(1, description="Number of GPUs")
    accumulate_grad_batches: int = Field(1, description="Gradient accumulation batches")
    val_check_interval: int = Field(ModelConstants.DEFAULT_VAL_CHECK_INTERVAL, description="Validation check interval")
    log_every_n_steps: int = Field(ModelConstants.DEFAULT_LOG_EVERY_N_STEPS, description="Log every n steps")
    save_top_k: int = Field(ModelConstants.DEFAULT_SAVE_TOP_K, description="Save top k models")
    limit_val_batches: int = Field(ModelConstants.DEFAULT_VALIDATION_BATCHES, description="Limit validation batches")
    num_sanity_val_steps: int = Field(
        ModelConstants.DEFAULT_SANITY_VAL_STEPS, description="Number of sanity validation steps"
    )
    every_n_train_steps: int = Field(
        ModelConstants.DEFAULT_CHECKPOINT_EVERY_N_STEPS, description="Checkpoint every n train steps"
    )

    # Data parameters
    dataset_name: str = Field("DATASET_NAME", description="Dataset name")
    num_workers: int = Field(4, description="Number of workers")
    pin_memory: bool = Field(True, description="Pin memory")
    bar_token_idx: int = Field(ModelConstants.BOS_TOKEN_ID, description="Bar token index")
    train_val_test_split: tuple = Field((0.7, 0.2, 0.1), description="Train/validation/test split")

    # Additional data loading flags
    max_bars_per_context: int = Field(-1, description="Maximum bars per context")
    max_contexts_per_file: int = Field(-1, description="Maximum contexts per file")
    bar_token_mask: Optional[str] = Field(None, description="Bar masking token")

    # Device and performance
    device: str = Field("cuda", description="Device to use")

    # Checkpoint configuration
    checkpoint_dir: str = Field("output/checkpoints/vae", description="Checkpoint directory")
    load_from_checkpoint: bool = Field(False, description="Whether to load from checkpoint")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint file")
    weights_path: Optional[str] = Field(None, description="Path to model weights")
    config_path: Optional[str] = Field(None, description="Path to model configuration")

    # Logging configuration
    use_wandb: bool = Field(False, description="Use Weights & Biases logging")


class LatentRepresentationParameters(BaseModel):
    """Parameters for generating latent representations."""

    # Model configuration
    checkpoint_path: str = Field(..., description="Path to trained VAE checkpoint")

    # Data parameters
    dataset_name: str = Field("DATASET_NAME", description="Dataset name")
    context_size: int = Field(-1, description="Context size (-1 for full context)")
    batch_size: int = Field(ModelConstants.DEFAULT_BATCH_SIZE, description="Batch size")
    num_workers: int = Field(4, description="Number of workers")
    pin_memory: bool = Field(True, description="Pin memory")
    max_positions: int = Field(ModelConstants.DEFAULT_MAX_POSITIONS, description="Maximum position embeddings")
    max_bars: int = Field(MAX_N_BARS, description="Maximum number of bars")

    # Processing parameters
    bar_token_idx: int = Field(ModelConstants.BOS_TOKEN_ID, description="Bar token index")
    train_val_test_split: tuple = Field((0.7, 0.2, 0.1), description="Train/validation/test split")

    # Additional data loading flags
    max_bars_per_context: int = Field(-1, description="Maximum bars per context")
    max_contexts_per_file: int = Field(-1, description="Maximum contexts per file")
    bar_token_mask: Optional[str] = Field(None, description="Bar masking token")

    # Device configuration
    device: str = Field("cuda", description="Device to use")

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
    num_sanity_val_steps: int = Field(
        ModelConstants.DEFAULT_SANITY_VAL_STEPS, description="Number of sanity validation steps"
    )
    save_top_k: int = Field(2, description="Save top k models")
    every_n_train_steps: int = Field(100, description="Checkpoint every n train steps")
