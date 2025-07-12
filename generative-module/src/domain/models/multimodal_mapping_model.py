from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


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
    temperature: float = Field(1.0, description="Temperature for sampling during generation")


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
    image_dim: int = Field(384, description="Image feature dimension")
    text_dim: int = Field(384, description="Text feature dimension")
    global_feature_dim: int = Field(64, description="Global feature dimension")
    global_feature_out_dim: int = Field(128, description="Global feature output dimension")
    mood_dim: int = Field(32, description="Mood feature dimension")
    mood_out_dim: int = Field(128, description="Mood feature output dimension")
    
    # Architecture parameters
    fusion_dim: int = Field(512, description="Fusion dimension")
    d_model: int = Field(512, description="Model dimension")
    latent_dim: int = Field(128, description="Latent dimension")
    output_dim: int = Field(256, description="Output dimension")
    context_size: int = Field(512, description="Context size")
    num_layers: int = Field(4, description="Number of layers")
    num_heads: int = Field(8, description="Number of attention heads")
    fusion_heads: int = Field(8, description="Number of fusion heads")
    fusion_type: str = Field("linear_concat", description="Fusion type")
    dropout: float = Field(0.1, description="Dropout rate")
    training: bool = Field(True, description="Training mode")
    
    # Modality dropout rates
    image_dropout_rate: float = Field(0.2, description="Image dropout rate")
    text_dropout_rate: float = Field(0.2, description="Text dropout rate")
    global_feature_dropout_rate: float = Field(0.2, description="Global feature dropout rate")
    mood_dropout_rate: float = Field(0.2, description="Mood dropout rate")
    
    # Learning rate and schedule
    lr: float = Field(5e-5, description="Learning rate")
    lr_schedule: str = Field("cosine", description="Learning rate schedule")
    warmup_steps: int = Field(2000, description="Warmup steps")
    max_steps: int = Field(100000, description="Maximum steps")
    
    # KL annealing
    use_kl_annealing: bool = Field(True, description="Use KL annealing")
    kl_start: float = Field(0.0, description="KL annealing start value")
    kl_end: float = Field(1.0, description="KL annealing end value")
    kl_anneal_steps: int = Field(10000, description="KL annealing steps")
    
    # Training parameters
    batch_size: int = Field(32, description="Batch size")
    max_epochs: int = Field(100, description="Maximum epochs")
    gpus: int = Field(1, description="Number of GPUs")
    accumulate_grad_batches: int = Field(1, description="Gradient accumulation batches")
    
    # Data parameters
    dataset_name: str = Field("ReMIDICaps", description="Dataset name")
    load_latent: bool = Field(True, description="Load latent representations")
    load_symb: bool = Field(True, description="Load symbolic features")
    load_emotions: bool = Field(True, description="Load emotions")
    load_global_features: bool = Field(True, description="Load global features")
    load_text_prompts: bool = Field(True, description="Load text prompts")
    load_images: bool = Field(True, description="Load images")
    
    # Device and performance
    device: str = Field("cuda", description="Device to use")
    num_workers: int = Field(4, description="Number of workers")
    pin_memory: bool = Field(True, description="Pin memory")
    
    # Checkpoint and logging
    checkpoint_dir: str = Field("output/checkpoints/multimodal_mapping", description="Checkpoint directory")
    log_every_n_steps: int = Field(50, description="Log every n steps")
    val_check_interval: int = Field(1000, description="Validation check interval")
    save_top_k: int = Field(2, description="Save top k models")
    
    # Validation parameters
    limit_val_batches: int = Field(64, description="Limit validation batches")
    num_sanity_val_steps: int = Field(2, description="Number of sanity validation steps") 