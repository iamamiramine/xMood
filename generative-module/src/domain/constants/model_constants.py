"""
Model Constants for PA-AI Generative Module

This module contains all model-related constants that were previously hardcoded
throughout the codebase. Centralizing these values improves maintainability
and makes the system more configurable.
"""

from enum import Enum
from typing import Dict, Any



class ModelConstants:
    """Central repository for model architecture constants"""
    
    # Mood Classification
    MOOD_DIMENSIONS = 9
    MOOD_NAMES = [
        "HAPPY_KEY", "DRAMATIC_KEY", "RELAXING_KEY", "LOVE_KEY", "DARK_KEY",
        "CHRISTMAS_KEY", "ENERGETIC_KEY", "MEDITATIVE_KEY", "MOTIVATIONAL_KEY"
    ]
    
    # Vocabulary and Token Constants
    DEFAULT_VOCAB_SIZE = 1382  # Based on actual REMI vocabulary size
    PAD_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    UNK_TOKEN_ID = 3
    MASK_TOKEN_ID = 4
    
    # Model Architecture Defaults
    DEFAULT_HIDDEN_DIM = 512
    DEFAULT_D_MODEL = 512
    DEFAULT_D_LATENT = 1024
    DEFAULT_EMBEDDING_SIZE = 300
    DEFAULT_CONTEXT_SIZE = 512
    DEFAULT_MAX_POSITIONS = 2048
    DEFAULT_MAX_BARS = 512
    
    # Text Processing
    MAX_TEXT_LENGTH = 77  # Standard CLIP text encoder length
    
    # Attention and Transformer Architecture
    DEFAULT_NUM_ATTENTION_HEADS = 8
    DEFAULT_ENCODER_LAYERS = 6
    DEFAULT_DECODER_LAYERS = 12
    DEFAULT_INTERMEDIATE_SIZE = 2048
    
    # Training Constants
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LEARNING_RATE = 1e-4
    DEFAULT_WEIGHT_DECAY = 1e-4
    DEFAULT_DROPOUT = 0.0
    
    # Data Processing
    DEFAULT_BUFFER_SIZE = 2048
    DEFAULT_VALIDATION_BATCHES = 64
    DEFAULT_SANITY_VAL_STEPS = 2
    
    # Multimodal Architecture
    DEFAULT_IMAGE_DIM = 384
    DEFAULT_TEXT_DIM = 384
    DEFAULT_FUSION_DIM = 512
    DEFAULT_LATENT_DIM = 128
    DEFAULT_OUTPUT_DIM = 256
    
    # Global Feature Dimensions
    DEFAULT_GLOBAL_FEATURE_DIM = 64
    DEFAULT_GLOBAL_FEATURE_OUT_DIM = 128
    DEFAULT_MOOD_DIM = 32
    DEFAULT_MOOD_OUT_DIM = 128
    
    # VQ-VAE Constants
    DEFAULT_N_CODES = 1024
    DEFAULT_N_GROUPS = 2
    DEFAULT_VQ_DECAY = 0.995
    DEFAULT_VQ_EPS = 1e-4
    DEFAULT_VQ_RESTART_THRESHOLD = 0.99
    
    # Training and Generation
    DEFAULT_MAX_LENGTH = 256
    DEFAULT_MAX_N_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.8
    DEFAULT_WARMUP_STEPS = 1000
    DEFAULT_MAX_STEPS = 10000
    
    # Checkpoint and Validation
    DEFAULT_SAVE_TOP_K = 3
    DEFAULT_CHECKPOINT_EVERY_N_STEPS = 500
    DEFAULT_VAL_CHECK_INTERVAL = 500
    DEFAULT_LOG_EVERY_N_STEPS = 100
