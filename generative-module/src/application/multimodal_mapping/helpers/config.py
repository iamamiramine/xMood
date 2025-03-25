"""Configuration module for the multimodal mapping service."""

# Default configuration for multimodal mapping service
DEFAULT_CONFIG = {
    # Model dimensions
    'text_dim': 384,            # Output dimension of text encoder (DistilBERT)
    'image_dim': 512,           # Output dimension of image encoder (CLIP)
    'structured_dim': 256,      # Output dimension of structured feature processor
    'structured_input_dim': 64, # Input dimension of structured features
    'fusion_dim': 512,          # Dimension of fused embedding
    'hidden_dim': 512,          # Hidden dimension of transformer models
    'piece_latent_dim': 256,    # Dimension of piece-level latent space
    'bar_latent_dim': 128,      # Dimension of bar-level latent space
    'output_dim': 256,          # Output dimension (split between symbolic and VQVAE features)
    
    # Transformer settings
    'num_layers': 4,            # Number of transformer layers
    'num_heads': 8,             # Number of attention heads
    'max_bars': 16,             # Maximum number of bars to generate
    'dropout': 0.1,             # Dropout rate
    
    # Fusion settings
    'fusion_type': 'cross_attention',  # 'cross_attention' or 'linear_concat'
    'fusion_heads': 8,          # Number of attention heads in cross-attention fusion
    
    # Training settings
    'learning_rate': 1e-4,      # Learning rate
    'batch_size': 16,           # Batch size
    'kl_weight': 0.1,           # Weight of KL divergence term
    'kl_annealing_steps': 10000, # Steps for KL annealing
    
    # Checkpoint settings
    'checkpoint_dir': 'checkpoints/multimodal_mapping',
    'model_name': 'multimodal_mapping_model',
}


def get_config(config_updates=None):
    """
    Get configuration dictionary with optional updates.
    
    Args:
        config_updates: Dictionary of configuration updates
        
    Returns:
        config: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_updates is not None:
        config.update(config_updates)
    
    return config 