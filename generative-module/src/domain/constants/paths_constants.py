import yaml
import os
import logging
import glob
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global variable to store the current config file path
_current_config_file: Optional[str] = None


def set_config_file(config_file_path: Optional[str] = None):
    """
    Set the configuration file to use for loading constants.
    
    Args:
        config_file_path: Path to the configuration file, or None to use latest
    """
    global _current_config_file
    _current_config_file = config_file_path
    logger.info(f"Set configuration file to: {config_file_path or 'latest'}")


def get_current_config_file() -> Optional[str]:
    """
    Get the currently set configuration file path.
    
    Returns:
        Path to the current configuration file, or None if using latest
    """
    return _current_config_file


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
    # If a specific config file is set, use it
    if _current_config_file:
        if os.path.exists(_current_config_file):
            logger.info(f"Using explicitly set configuration file: {_current_config_file}")
            return _current_config_file
        else:
            logger.warning(f"Explicitly set config file not found: {_current_config_file}")
    
    # Otherwise, find the latest config file
    latest_file = find_latest_config_file()
    if latest_file:
        logger.info(f"Using latest configuration file: {latest_file}")
        return latest_file
    
    logger.warning("No configuration file available")
    return None


def list_available_configs() -> list:
    """
    List all available configuration files in the shared/config directory.
    
    Returns:
        List of configuration file paths, sorted by modification time (newest first)
    """
    config_dir = os.path.join("shared", "config")
    
    if not os.path.exists(config_dir):
        logger.warning(f"Configuration directory not found: {config_dir}")
        return []
    
    pattern = os.path.join(config_dir, "*.yaml")
    config_files = glob.glob(pattern)
    
    if not config_files:
        logger.warning(f"No configuration files found in {config_dir}")
        return []
    
    # Sort by modification time (newest first)
    config_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    logger.info(f"Found {len(config_files)} configuration files")
    return config_files


def load_yaml_config_file(file_path: str, description: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file with proper error handling.
    
    Args:
        file_path: Path to the YAML configuration file
        description: Human-readable description of the configuration file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the configuration file contains invalid YAML
        Exception: For other configuration loading errors
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{description} file not found at: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        if not isinstance(config_data, dict):
            raise ValueError(f"{description} file must contain a YAML object, got {type(config_data)}")
        
        logger.info(f"Successfully loaded {description} from {file_path}")
        return config_data
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in {description} file at {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading {description} from {file_path}: {e}")
        raise


def get_paths_config() -> Dict[str, Any]:
    """Get paths configuration from the specified or latest YAML config file with fallback values."""
    
    # Get the config file to use
    config_file = get_config_file_path()
    
    if config_file:
        try:
            config = load_yaml_config_file(config_file, "unified configuration")
            paths_config = config.get("paths", {})
            
            if paths_config:
                logger.info(f"Loaded paths configuration from {config_file}")
                return paths_config
            else:
                logger.warning(f"No 'paths' section found in {config_file}")
                
        except Exception as e:
            logger.error(f"Failed to load paths configuration from {config_file}: {e}")
    
    # Fallback to default paths
    logger.warning("Using default paths configuration")
    return {
        "ROOT_OUTPUT": "output",
        "DATASETS_PATH": "datasets",
        "DATASET_NAME": "EMOPIA",
    }


def get_main_config() -> Dict[str, Any]:
    """Get main configuration from the specified or latest YAML config file with fallback values."""
    
    # Get the config file to use
    config_file = get_config_file_path()
    
    if config_file:
        try:
            config = load_yaml_config_file(config_file, "unified configuration")
            logger.info(f"Loaded main configuration from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load main configuration from {config_file}: {e}")
    
    # Fallback to default config
    logger.warning("Using default main configuration")
    return {
        "global": {
            "device": "cuda",
            "seed": 42,
            "debug": False,
            "output_root": "output/",
            "checkpoints_root": "checkpoints/",
            "logs_root": "logs/"
        },
        "paths": {
            "ROOT_OUTPUT": "output",
            "DATASETS_PATH": "datasets",
            "DATASET_NAME": "EMOPIA"
        },
        "midi": {
            "pos_per_quarter": 12,
            "resolution": 480,
            "max_bar_length": 3,
            "max_bars": 512,
            "pad_idx": 338
        }
    }


# Load configurations with proper error handling
try:
    paths = get_paths_config()
    config = get_main_config()
except Exception as e:
    logger.critical(f"Critical error loading configuration: {e}")
    # Set minimal fallback values to prevent import errors
    paths = {"ROOT_OUTPUT": "output", "DATASETS_PATH": "datasets", "DATASET_NAME": "EMOPIA"}
    config = {
        "global": {"device": "cuda", "seed": 42},
        "paths": {"ROOT_OUTPUT": "output", "DATASETS_PATH": "datasets", "DATASET_NAME": "EMOPIA"},
        "midi": {"max_bars": 512}
    }

def get_config_value(config_dict: Dict[str, Any], key_path: str, default_value: Any = None) -> Any:
    """
    Safely get a nested configuration value with fallback.
    
    Args:
        config_dict: Configuration dictionary
        key_path: Dot-separated path to the configuration key (e.g., "paths.ROOT_OUTPUT")
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
DATASET_NAME = get_config_value(paths, "DATASET_NAME", "EMOPIA")

# Construct paths using safe configuration values
MIDI_PATH = f"{DATASETS_PATH}/{DATASET_NAME}/midi/"
LABELS_PATH = f"{DATASETS_PATH}/{DATASET_NAME}/labels/"

PROCESSED_PATH = f"{ROOT_OUTPUT}/processed/"
GENERATED_PATH = f"{ROOT_OUTPUT}/generated/"
LATENTS_PATH = f"{ROOT_OUTPUT}/latents/"

# Log the loaded configuration for debugging
logger.info(f"Loaded path constants - ROOT_OUTPUT: {ROOT_OUTPUT}, DATASETS_PATH: {DATASETS_PATH}, DATASET_NAME: {DATASET_NAME}")
