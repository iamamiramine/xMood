from pydantic import BaseModel, Field
from typing import Optional


class GenerateFromMIDIParameters(BaseModel):
    """Parameters for generating MIDI from existing MIDI files."""

    device: str
    latent_midi: str
    symbolic_midi: str
    emotions_midi: str
    output_folder: str
    output_name: str
    load_from_checkpoint: bool
    load_weights: bool
    checkpoint_path: str
    weights_path: str
    config_path: str
    context_size: int = 256
    max_bars: int = 16
    max_positions: int = 512
    max_n_tokens: int = 1024
    temperature: float = 0.8
    initial_context: int = 256


class GeneratorTrainingParameters(BaseModel):
    """Parameters for training generator models."""
    
    # Model architecture
    d_model: int = Field(512, description="Model dimension")
    context_size: int = Field(512, description="Context size")
    batch_size: int = Field(32, description="Batch size")
    num_attention_heads: int = Field(8, description="Number of attention heads")
    d_latent: int = Field(128, description="Latent dimension")
    max_bars: int = Field(16, description="Maximum bars")
    max_positions: int = Field(512, description="Maximum positions")
    encoder_layers: int = Field(6, description="Number of encoder layers")
    decoder_layers: int = Field(6, description="Number of decoder layers")
    intermediate_size: int = Field(2048, description="Intermediate size")
    vocab_size: int = Field(1382, description="Vocabulary size")
    dropout: float = Field(0.1, description="Dropout rate")
    
    # Training parameters
    lr: float = Field(1e-4, description="Learning rate")
    lr_schedule: str = Field("sqrt_decay", description="Learning rate schedule")
    warmup_steps: int = Field(4000, description="Warmup steps")
    max_steps: int = Field(100000, description="Maximum steps")
    max_epochs: int = Field(100, description="Maximum epochs")
    weight_decay: float = Field(1e-4, description="Weight decay")
    
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
    encode: bool = Field(False, description="Encode data")
    
    # Device and performance
    device: str = Field("cuda", description="Device to use")
    
    # Checkpoint configuration
    checkpoint_dir: str = Field("output/checkpoints/generator", description="Checkpoint directory")
    load_from_checkpoint: bool = Field(False, description="Whether to load from checkpoint")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint file")
    weights_path: Optional[str] = Field(None, description="Path to model weights")
    config_path: Optional[str] = Field(None, description="Path to model configuration")
