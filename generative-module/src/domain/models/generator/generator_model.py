from pydantic import BaseModel, Field
from typing import Optional

# Import constants
from domain.constants.model_constants import ModelConstants


class GenerateFromMIDIParameters(BaseModel):
    """Parameters for generating MIDI from existing MIDI files."""

    device: str
    latent_midi: str
    symbolic_midi: str
    output_folder: str
    output_name: str
    load_from_checkpoint: bool
    load_weights: bool
    checkpoint_path: str
    weights_path: str
    config_path: str
    context_size: int = ModelConstants.DEFAULT_CONTEXT_SIZE
    max_bars: int = 16
    max_positions: int = ModelConstants.DEFAULT_MAX_POSITIONS
    max_n_tokens: int = ModelConstants.DEFAULT_MAX_N_TOKENS
    temperature: float = ModelConstants.DEFAULT_TEMPERATURE
    initial_context: int = ModelConstants.DEFAULT_CONTEXT_SIZE


class GeneratorTrainingParameters(BaseModel):
    """Parameters for training generator models."""
    
    # Model architecture
    d_model: int = Field(ModelConstants.DEFAULT_D_MODEL, description="Model dimension")
    context_size: int = Field(ModelConstants.DEFAULT_CONTEXT_SIZE, description="Context size")
    batch_size: int = Field(ModelConstants.DEFAULT_BATCH_SIZE, description="Batch size")
    num_attention_heads: int = Field(ModelConstants.DEFAULT_NUM_ATTENTION_HEADS, description="Number of attention heads")
    d_latent: int = Field(ModelConstants.DEFAULT_LATENT_DIM, description="Latent dimension")
    max_bars: int = Field(16, description="Maximum bars")
    max_positions: int = Field(ModelConstants.DEFAULT_MAX_POSITIONS, description="Maximum positions")
    encoder_layers: int = Field(ModelConstants.DEFAULT_ENCODER_LAYERS, description="Number of symbolic layers")
    decoder_layers: int = Field(ModelConstants.DEFAULT_DECODER_LAYERS, description="Number of decoder layers")
    intermediate_size: int = Field(ModelConstants.DEFAULT_INTERMEDIATE_SIZE, description="Intermediate size")
    vocab_size: int = Field(ModelConstants.DEFAULT_VOCAB_SIZE, description="Vocabulary size")
    dropout: float = Field(ModelConstants.DEFAULT_DROPOUT, description="Dropout rate")
    
    # Training parameters
    lr: float = Field(ModelConstants.DEFAULT_LEARNING_RATE, description="Learning rate")
    lr_schedule: str = Field("sqrt_decay", description="Learning rate schedule")
    warmup_steps: int = Field(ModelConstants.DEFAULT_WARMUP_STEPS, description="Warmup steps")
    max_steps: int = Field(ModelConstants.DEFAULT_MAX_STEPS, description="Maximum steps")
    max_epochs: int = Field(100, description="Maximum epochs")
    weight_decay: float = Field(ModelConstants.DEFAULT_WEIGHT_DECAY, description="Weight decay")
    
    # Training configuration
    gpus: int = Field(1, description="Number of GPUs")
    accumulate_grad_batches: int = Field(1, description="Gradient accumulation batches")
    val_check_interval: int = Field(ModelConstants.DEFAULT_VAL_CHECK_INTERVAL, description="Validation check interval")
    log_every_n_steps: int = Field(ModelConstants.DEFAULT_LOG_EVERY_N_STEPS, description="Log every n steps")
    save_top_k: int = Field(ModelConstants.DEFAULT_SAVE_TOP_K, description="Save top k models")
    limit_val_batches: int = Field(ModelConstants.DEFAULT_VALIDATION_BATCHES, description="Limit validation batches")
    num_sanity_val_steps: int = Field(ModelConstants.DEFAULT_SANITY_VAL_STEPS, description="Number of sanity validation steps")
    every_n_train_steps: int = Field(1000, description="Checkpoint every n train steps")
    
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
    checkpoint_dir: str = Field("output/checkpoints/generator", description="Checkpoint directory")
    load_from_checkpoint: bool = Field(False, description="Whether to load from checkpoint")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint file")
    weights_path: Optional[str] = Field(None, description="Path to model weights")
    config_path: Optional[str] = Field(None, description="Path to model configuration")
    load_weights: bool = Field(False, description="Load weights instead of full checkpoint")
