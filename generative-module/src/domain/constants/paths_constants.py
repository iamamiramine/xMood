import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config_file(file_path: str, description: str) -> Dict[str, Any]:
    """
    Load and parse a JSON configuration file with proper error handling.
    
    Args:
        file_path: Path to the JSON configuration file
        description: Human-readable description of the configuration file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        json.JSONDecodeError: If the configuration file contains invalid JSON
        Exception: For other configuration loading errors
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{description} file not found at: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            config_data = json.load(file)
        
        if not isinstance(config_data, dict):
            raise ValueError(f"{description} file must contain a JSON object, got {type(config_data)}")
        
        logger.info(f"Successfully loaded {description} from {file_path}")
        return config_data
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {description} file at {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading {description} from {file_path}: {e}")
        raise


def get_paths_config() -> Dict[str, Any]:
    """Get paths configuration with fallback values."""
    paths_file = os.path.join("shared", "config", "paths.json")
    
    try:
        return load_config_file(paths_file, "paths configuration")
    except FileNotFoundError:
        # Fallback to default paths
        logger.warning(f"Paths configuration file not found, using default values")
        return {
            "ROOT_OUTPUT": "output",
            "DATASETS_PATH": "datasets",
        }
    except Exception as e:
        logger.error(f"Failed to load paths configuration: {e}")
        # Return minimal fallback configuration
        return {
            "ROOT_OUTPUT": "output",
            "DATASETS_PATH": "datasets",
        }


def get_main_config() -> Dict[str, Any]:
    """Get main configuration with fallback values."""
    config_file = os.path.join("shared", "config", "config.json")
    
    try:
        return load_config_file(config_file, "main configuration")
    except FileNotFoundError:
        # Fallback to default config
        logger.warning(f"Main configuration file not found, using default values")
        return {
            "dataloader": {
                "dataset_name": "EMOPIA",
                "max_bars": 512,
            }
        }
    except Exception as e:
        logger.error(f"Failed to load main configuration: {e}")
        # Return minimal fallback configuration
        return {
            "dataloader": {
                "dataset_name": "EMOPIA",
                "max_bars": 512,
            }
        }


# Load configurations with proper error handling
try:
    paths = get_paths_config()
    config = get_main_config()
except Exception as e:
    logger.critical(f"Critical error loading configuration: {e}")
    # Set minimal fallback values to prevent import errors
    paths = {"ROOT_OUTPUT": "output", "DATASETS_PATH": "datasets"}
    config = {"dataloader": {"dataset_name": "EMOPIA", "max_bars": 512}}

def get_config_value(config_dict: Dict[str, Any], key_path: str, default_value: Any = None) -> Any:
    """
    Safely get a nested configuration value with fallback.
    
    Args:
        config_dict: Configuration dictionary
        key_path: Dot-separated path to the configuration key (e.g., "dataloader.dataset_name")
        default_value: Default value if key is not found
        
    Returns:
        Configuration value or default value
    """
    try:
        keys = key_path.split(".")
        value = config_dict
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError, AttributeError):
        logger.warning(f"Configuration key '{key_path}' not found, using default value: {default_value}")
        return default_value


# Extract configuration values with fallbacks
ROOT_OUTPUT = get_config_value(paths, "ROOT_OUTPUT", "output")
DATASETS_PATH = get_config_value(paths, "DATASETS_PATH", "datasets")

DATASET_NAME = get_config_value(config, "dataloader.dataset_name", "EMOPIA")

# Construct paths using safe configuration values
MIDI_PATH = f"{DATASETS_PATH}/{DATASET_NAME}/midi/"
LABELS_PATH = f"{DATASETS_PATH}/{DATASET_NAME}/labels/"

PROCESSED_PATH = f"{ROOT_OUTPUT}/processed/"
CHECKPOINTS_PATH = f"{ROOT_OUTPUT}/checkpoints/"
GENERATED_PATH = f"{ROOT_OUTPUT}/generated/"
LATENTS_PATH = f"{ROOT_OUTPUT}/latents/"

CHORDS_PATH = f"{ROOT_OUTPUT}/chords/"
KEYS_PATH = f"{ROOT_OUTPUT}/keys/"

ENCODINGS_PATH = f"{ROOT_OUTPUT}/encodings/"

SYMBOLIC_FEATURES_PATH = f"{ROOT_OUTPUT}/symbolic_features/"

REPRESENTATIONS_PATH = f"{ROOT_OUTPUT}/representations/"

VAE_PATH = f"{ROOT_OUTPUT}/VQVAE/"
CODES_PATH = f"{ROOT_OUTPUT}/codes/"
FEATURES_PATH = f"{ROOT_OUTPUT}/features/"

EMOTION_MAPPING_PATH = f"{ROOT_OUTPUT}/emotion_mapping/"

GENERATOR_PATH = f"{ROOT_OUTPUT}/generator/"

# Log the loaded configuration for debugging
logger.info(f"Loaded path constants - ROOT_OUTPUT: {ROOT_OUTPUT}, DATASETS_PATH: {DATASETS_PATH}, DATASET_NAME: {DATASET_NAME}")
