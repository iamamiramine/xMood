"""
Model Constants for PA-AI Generative Module

This module contains all model-related constants that were previously hardcoded
throughout the codebase. Centralizing these values improves maintainability
and makes the system more configurable.
"""

import os
import yaml
import logging
import glob
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def find_latest_config_file() -> Optional[str]:
    """
    Find the latest configuration file in the shared/config directory.
    
    Returns:
        Path to the latest configuration file, or None if no files found
    """
    config_dir = os.path.join("shared", "config")
    
    if not os.path.exists(config_dir):
        logger.warning(f"Configuration directory not found: {config_dir}")
        return None
    
    # Look for YAML files with the new naming pattern
    pattern = os.path.join(config_dir, "*.yaml")
    config_files = glob.glob(pattern)
    
    if not config_files:
        logger.warning(f"No configuration files found in {config_dir}")
        return None
    
    # Sort by modification time (newest first)
    config_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    latest_file = config_files[0]
    logger.info(f"Found latest configuration file: {latest_file}")
    return latest_file


def get_config_file_path() -> Optional[str]:
    """
    Get the configuration file path to use.
    Priority: 1. Explicitly set config file, 2. Latest config file
    
    Returns:
        Path to the configuration file to use, or None if no config available
    """
    # Import the config file selection function from paths_constants
    try:
        from domain.constants.paths_constants import get_config_file_path as get_paths_config_file
        return get_paths_config_file()
    except ImportError:
        # Fallback to finding latest config file
        return find_latest_config_file()


def load_model_config() -> Dict[str, Any]:
    """
    Load model configuration from the specified or latest YAML config file.
    
    Returns:
        Dictionary containing model configuration values
        
    Raises:
        FileNotFoundError: If no configuration file is found
        Exception: For other configuration loading errors
    """
    config_path = get_config_file_path()
    
    if not config_path:
        raise FileNotFoundError("No configuration file found")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise ValueError(f"Configuration file must contain a YAML object, got {type(config)}")
        
        # Extract model-related configurations from service configs
        model_config = {}
        
        # Get global configuration
        global_config = config.get("global", {})
        model_config.update(global_config)
        
        # Extract common parameters from service configurations
        for service_name in ["symbolic_service", "latent_service", "generator_service", "multimodal_service"]:
            service_config = config.get(service_name, {})
            if service_config:
                # Look for common parameters in service functions
                for func_name, func_config in service_config.items():
                    if isinstance(func_config, dict):
                        # Extract common model parameters
                        common_params = [
                            "device", "batch_size", "context_size", "dataset_name", 
                            "num_workers", "pin_memory", "max_bars", "max_positions",
                            "d_model", "dropout", "lr", "max_steps", "warmup_steps"
                        ]
                        for param in common_params:
                            if param in func_config and param not in model_config:
                                model_config[param] = func_config[param]
        
        logger.info(f"Successfully loaded model configuration from {config_path}")
        return model_config
        
    except Exception as e:
        logger.error(f"Error loading model configuration from {config_path}: {e}")
        raise


def get_model_config_safe() -> Dict[str, Any]:
    """
    Load model configuration with fallback values.
    
    Returns:
        Dictionary containing model configuration values with fallbacks
    """
    try:
        return load_model_config()
    except Exception as e:
        logger.warning(f"Failed to load model configuration, using defaults: {e}")
        return {}


# Load model configuration with fallback
try:
    model_config = get_model_config_safe()
except Exception as e:
    logger.critical(f"Critical error loading model configuration: {e}")
    model_config = {}


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
    
    # Model Architecture Defaults - Load from config if available, otherwise use defaults
    DEFAULT_HIDDEN_DIM = 512
    DEFAULT_D_MODEL = model_config.get("d_model", 512)
    DEFAULT_D_LATENT = 1024
    DEFAULT_EMBEDDING_SIZE = 300
    DEFAULT_CONTEXT_SIZE = model_config.get("context_size", 512)
    DEFAULT_MAX_POSITIONS = model_config.get("max_positions", 2048)
    DEFAULT_MAX_BARS = model_config.get("max_bars", 512)
    
    # Text Processing
    MAX_TEXT_LENGTH = 77  # Standard CLIP text symbolic length
    
    # Attention and Transformer Architecture
    DEFAULT_NUM_ATTENTION_HEADS = 8
    DEFAULT_ENCODER_LAYERS = 6
    DEFAULT_DECODER_LAYERS = 12
    DEFAULT_INTERMEDIATE_SIZE = 2048
    
    # Training Constants - Load from config if available, otherwise use defaults
    DEFAULT_BATCH_SIZE = model_config.get("batch_size", 32)
    DEFAULT_LEARNING_RATE = model_config.get("lr", 1e-4)
    DEFAULT_WEIGHT_DECAY = 1e-4
    DEFAULT_DROPOUT = model_config.get("dropout", 0.0)
    
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
    
    # Training and Generation - Load from config if available, otherwise use defaults
    DEFAULT_MAX_LENGTH = 256
    DEFAULT_MAX_N_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.8
    DEFAULT_WARMUP_STEPS = model_config.get("warmup_steps", 1000)
    DEFAULT_MAX_STEPS = model_config.get("max_steps", 10000)
    
    # Checkpoint and Validation
    DEFAULT_SAVE_TOP_K = 3
    DEFAULT_CHECKPOINT_EVERY_N_STEPS = 500
    DEFAULT_VAL_CHECK_INTERVAL = 500
    DEFAULT_LOG_EVERY_N_STEPS = 100

# Log the loaded configuration for debugging
logger.info(f"Loaded model constants - d_model: {ModelConstants.DEFAULT_D_MODEL}, context_size: {ModelConstants.DEFAULT_CONTEXT_SIZE}, batch_size: {ModelConstants.DEFAULT_BATCH_SIZE}")
