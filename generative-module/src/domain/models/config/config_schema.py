"""
Configuration Schema for PA-AI Generative Module

This module defines Pydantic models for configuration validation and type safety.
All configuration parameters should be defined here with proper types, defaults,
and documentation.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from domain.constants.model_constants import ModelConstants, ClassifierType


class BaseConfig(BaseModel):
    """Base configuration with common parameters"""
    
    class Config:
        # Allow arbitrary types for enum compatibility
        arbitrary_types_allowed = True
        # Use enum values in JSON serialization
        use_enum_values = True


class ModelConfig(BaseConfig):
    """Base model architecture configuration"""
    
    d_model: int = Field(
        default=ModelConstants.DEFAULT_D_MODEL,
        description="Hidden dimension size for transformer models",
        ge=64, le=2048
    )
    context_size: int = Field(
        default=ModelConstants.DEFAULT_CONTEXT_SIZE,
        description="Maximum context length for sequences",
        ge=64, le=4096
    )
    batch_size: int = Field(
        default=ModelConstants.DEFAULT_BATCH_SIZE,
        description="Training batch size",
        ge=1, le=512
    )
    num_attention_heads: int = Field(
        default=ModelConstants.DEFAULT_NUM_ATTENTION_HEADS,
        description="Number of attention heads",
        ge=1, le=32
    )
    
    @validator('d_model')
    def d_model_must_be_divisible_by_heads(cls, v, values):
        """Ensure d_model is divisible by number of attention heads"""
        if 'num_attention_heads' in values:
            if v % values['num_attention_heads'] != 0:
                raise ValueError('d_model must be divisible by num_attention_heads')
        return v


class ClassifierConfig(ModelConfig):
    """Configuration for classifier models"""
    
    cls_type: ClassifierType = Field(
        default=ClassifierType.MOOD,
        description="Type of classifier (MOOD, AV, A, V)"
    )
    num_of_dim: int = Field(
        default=ModelConstants.MOOD_DIMENSIONS,
        description="Output dimensions for classifier",
        ge=1, le=100
    )
    lstm_hidden_dim: int = Field(
        default=128,
        description="LSTM hidden dimension size",
        ge=32, le=1024
    )
    embedding_size: int = Field(
        default=ModelConstants.DEFAULT_EMBEDDING_SIZE,
        description="Embedding layer dimension",
        ge=64, le=1024
    )
    vocab_size: int = Field(
        default=ModelConstants.DEFAULT_VOCAB_SIZE,
        description="Vocabulary size",
        ge=100, le=10000
    )
    r: int = Field(
        default=14,
        description="Number of attention aspects",
        ge=1, le=50
    )
    da: int = Field(
        default=128,
        description="Attention hidden dimension",
        ge=32, le=512
    )
    # Note: max_context_size is now standardized as context_size in the base ModelConfig
    # This field is kept for backward compatibility but context_size is the preferred standard


class GeneratorConfig(ModelConfig):
    """Configuration for generator models"""
    
    d_latent: int = Field(
        default=ModelConstants.DEFAULT_D_LATENT,
        description="Latent dimension size",
        ge=128, le=2048
    )
    max_bars: int = Field(
        default=ModelConstants.DEFAULT_MAX_BARS,
        description="Maximum number of bars",
        ge=1, le=1024
    )
    max_positions: int = Field(
        default=ModelConstants.DEFAULT_MAX_POSITIONS,
        description="Maximum position embeddings",
        ge=512, le=8192
    )
    encoder_layers: int = Field(
        default=ModelConstants.DEFAULT_ENCODER_LAYERS,
        description="Number of encoder layers",
        ge=1, le=24
    )
    decoder_layers: int = Field(
        default=ModelConstants.DEFAULT_DECODER_LAYERS,
        description="Number of decoder layers",
        ge=1, le=24
    )
    intermediate_size: int = Field(
        default=ModelConstants.DEFAULT_INTERMEDIATE_SIZE,
        description="Feed-forward intermediate size",
        ge=512, le=8192
    )
    vocab_size: int = Field(
        default=ModelConstants.DEFAULT_VOCAB_SIZE,
        description="Vocabulary size",
        ge=100, le=10000
    )


class MultimodalConfig(ModelConfig):
    """Configuration for multimodal mapping models"""
    
    image_dim: int = Field(
        default=ModelConstants.DEFAULT_IMAGE_DIM,
        description="Image encoder output dimension",
        ge=128, le=1024
    )
    text_dim: int = Field(
        default=ModelConstants.DEFAULT_TEXT_DIM,
        description="Text encoder output dimension",
        ge=128, le=1024
    )
    global_feature_dim: int = Field(
        default=ModelConstants.DEFAULT_GLOBAL_FEATURE_DIM,
        description="Global feature input dimension",
        ge=16, le=512
    )
    global_feature_out_dim: int = Field(
        default=ModelConstants.DEFAULT_GLOBAL_FEATURE_OUT_DIM,
        description="Global feature output dimension",
        ge=32, le=512
    )
    mood_dim: int = Field(
        default=ModelConstants.DEFAULT_MOOD_DIM,
        description="Mood feature input dimension",
        ge=8, le=128
    )
    mood_out_dim: int = Field(
        default=ModelConstants.DEFAULT_MOOD_OUT_DIM,
        description="Mood feature output dimension",
        ge=32, le=512
    )
    fusion_dim: int = Field(
        default=ModelConstants.DEFAULT_FUSION_DIM,
        description="Fusion layer dimension",
        ge=256, le=1024
    )
    # Note: hidden_dim is now standardized as d_model in the base ModelConfig
    # This field is kept for backward compatibility but d_model is the preferred standard
    latent_dim: int = Field(
        default=ModelConstants.DEFAULT_LATENT_DIM,
        description="VAE latent dimension",
        ge=64, le=512
    )
    output_dim: int = Field(
        default=ModelConstants.DEFAULT_OUTPUT_DIM,
        description="Final output dimension",
        ge=128, le=1024
    )
    num_layers: int = Field(
        default=4,
        description="Number of transformer layers",
        ge=1, le=12
    )
    fusion_type: Literal["cross_attention", "linear_concat"] = Field(
        default="linear_concat",
        description="Type of fusion mechanism"
    )
    fusion_heads: int = Field(
        default=8,
        description="Number of fusion attention heads",
        ge=1, le=16
    )


class VAEConfig(ModelConfig):
    """Configuration for VAE models"""
    
    n_codes: int = Field(
        default=ModelConstants.DEFAULT_N_CODES,
        description="Number of codebook entries",
        ge=256, le=8192
    )
    n_groups: int = Field(
        default=ModelConstants.DEFAULT_N_GROUPS,
        description="Number of codebook groups",
        ge=1, le=16
    )
    d_latent: int = Field(
        default=ModelConstants.DEFAULT_D_LATENT,
        description="Latent space dimension",
        ge=256, le=2048
    )
    encoder_layers: int = Field(
        default=ModelConstants.DEFAULT_ENCODER_LAYERS,
        description="Number of encoder layers",
        ge=1, le=24
    )
    decoder_layers: int = Field(
        default=ModelConstants.DEFAULT_DECODER_LAYERS,
        description="Number of decoder layers",
        ge=1, le=24
    )
    encoder_ffn_dim: int = Field(
        default=ModelConstants.DEFAULT_INTERMEDIATE_SIZE,
        description="Encoder feed-forward dimension",
        ge=512, le=8192
    )
    decoder_ffn_dim: int = Field(
        default=ModelConstants.DEFAULT_INTERMEDIATE_SIZE,
        description="Decoder feed-forward dimension",
        ge=512, le=8192
    )
    disable_vq: bool = Field(
        default=False,
        description="Disable vector quantization (use regular VAE)"
    )
    beta: float = Field(
        default=0.02,
        description="VQ loss weight",
        ge=0.0, le=1.0
    )
    decay: float = Field(
        default=ModelConstants.DEFAULT_VQ_DECAY,
        description="EMA decay rate",
        ge=0.9, le=0.999
    )
    eps: float = Field(
        default=ModelConstants.DEFAULT_VQ_EPS,
        description="EMA epsilon",
        ge=1e-6, le=1e-3
    )
    restart_threshold: float = Field(
        default=ModelConstants.DEFAULT_VQ_RESTART_THRESHOLD,
        description="Codebook restart threshold",
        ge=0.9, le=1.0
    )


class TrainingConfig(BaseConfig):
    """Training configuration"""
    
    lr: float = Field(
        default=ModelConstants.DEFAULT_LEARNING_RATE,
        description="Learning rate",
        ge=1e-6, le=1e-2
    )
    weight_decay: float = Field(
        default=ModelConstants.DEFAULT_WEIGHT_DECAY,
        description="Weight decay for optimizer",
        ge=0.0, le=1e-2
    )
    warmup_steps: int = Field(
        default=ModelConstants.DEFAULT_WARMUP_STEPS,
        description="Number of warmup steps",
        ge=0, le=10000
    )
    max_steps: int = Field(
        default=ModelConstants.DEFAULT_MAX_STEPS,
        description="Maximum training steps",
        ge=1000, le=1000000
    )
    max_epochs: int = Field(
        default=100,
        description="Maximum training epochs",
        ge=1, le=1000
    )
    val_check_interval: int = Field(
        default=ModelConstants.DEFAULT_VAL_CHECK_INTERVAL,
        description="Validation check interval",
        ge=10, le=5000
    )
    log_every_n_steps: int = Field(
        default=ModelConstants.DEFAULT_LOG_EVERY_N_STEPS,
        description="Logging frequency",
        ge=1, le=1000
    )
    checkpoint_every_n_steps: int = Field(
        default=ModelConstants.DEFAULT_CHECKPOINT_EVERY_N_STEPS,
        description="Checkpoint saving frequency",
        ge=100, le=10000
    )
    save_top_k: int = Field(
        default=ModelConstants.DEFAULT_SAVE_TOP_K,
        description="Number of best checkpoints to keep",
        ge=1, le=10
    )
    limit_val_batches: int = Field(
        default=ModelConstants.DEFAULT_VALIDATION_BATCHES,
        description="Limit validation batches",
        ge=1, le=1000
    )
    num_sanity_val_steps: int = Field(
        default=ModelConstants.DEFAULT_SANITY_VAL_STEPS,
        description="Number of sanity validation steps",
        ge=0, le=10
    )
    lr_schedule: Literal["const", "sqrt_decay", "cosine", "linear"] = Field(
        default="const",
        description="Learning rate schedule type"
    )


class DataConfig(BaseConfig):
    """Data loading and processing configuration"""
    
    dataset_name: str = Field(
        default="ReMIDICaps",
        description="Name of the dataset"
    )
    max_bars_per_context: int = Field(
        default=-1,
        description="Maximum bars per context (-1 for no limit)",
        ge=-1, le=512
    )
    max_contexts_per_file: int = Field(
        default=-1,
        description="Maximum contexts per file (-1 for no limit)",
        ge=-1, le=100
    )
    num_workers: int = Field(
        default=2,
        description="Number of data loading workers",
        ge=0, le=16
    )
    pin_memory: bool = Field(
        default=True,
        description="Pin memory for faster GPU transfer"
    )
    load_latent: bool = Field(
        default=True,
        description="Load latent features"
    )
    load_symb: bool = Field(
        default=True,
        description="Load symbolic features"
    )
    load_emotions: bool = Field(
        default=True,
        description="Load emotion features"
    )
    load_global_features: bool = Field(
        default=False,
        description="Load global features"
    )
    load_text_prompts: bool = Field(
        default=False,
        description="Load text prompts"
    )
    train_val_test_split: List[float] = Field(
        default=[0.7, 0.2, 0.1],
        description="Train/validation/test split ratios"
    )
    
    @validator('train_val_test_split')
    def split_must_sum_to_one(cls, v):
        """Ensure split ratios sum to 1.0"""
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError('train_val_test_split must sum to 1.0')
        return v


class FullConfig(BaseConfig):
    """Complete configuration combining all components"""
    
    # Model configurations
    classifier: Optional[ClassifierConfig] = Field(default_factory=ClassifierConfig)
    generator: Optional[GeneratorConfig] = Field(default_factory=GeneratorConfig)
    multimodal_mapping: Optional[MultimodalConfig] = Field(default_factory=MultimodalConfig)
    vae: Optional[VAEConfig] = Field(default_factory=VAEConfig)
    
    # Training and data configurations
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    
    # Global settings
    device: str = Field(
        default="cuda",
        description="Device to use for training/inference"
    )
    seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    
    def get_config_for_service(self, service_name: str) -> Optional[BaseConfig]:
        """Get configuration for a specific service"""
        service_configs = {
            "classifier": self.classifier,
            "generator": self.generator,
            "multimodal_mapping": self.multimodal_mapping,
            "vae": self.vae,
            "training": self.training,
            "data": self.data,
        }
        return service_configs.get(service_name) 