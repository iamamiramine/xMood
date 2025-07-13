import os
import yaml
import numpy as np
import logging
import glob
from typing import Dict, Any, Tuple, Optional

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


def load_midi_config() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load MIDI configuration from the specified or latest YAML config file with proper error handling.
    
    Returns:
        Tuple of (midi_config, global_config) dictionaries
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the configuration file contains invalid YAML
        Exception: For other configuration loading errors
    """
    config_path = get_config_file_path()
    
    if not config_path:
        raise FileNotFoundError("No configuration file found")
    
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise ValueError(f"Configuration file must contain a YAML object, got {type(config)}")
        
        midi_config = config.get("midi", {})
        global_config = config.get("global", {})
        
        logger.info(f"Successfully loaded MIDI configuration from {config_path}")
        return midi_config, global_config
        
    except FileNotFoundError as e:
        logger.error(f"MIDI configuration file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in MIDI configuration file at {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading MIDI configuration from {config_path}: {e}")
        raise


def get_midi_config_safe() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load MIDI configuration from YAML with fallback values.
    
    Returns:
        Tuple of (midi_config, global_config) dictionaries with fallback values
    """
    try:
        return load_midi_config()
    except FileNotFoundError:
        logger.warning("MIDI configuration file not found, using default values")
        return {
            "pos_per_quarter": 12,
            "resolution": 480,
            "max_bar_length": 3,
            "max_bars": 512,
            "pad_idx": 338
        }, {
            "device": "cuda",
            "seed": 42,
            "debug": False,
            "output_root": "output/",
            "checkpoints_root": "checkpoints/",
            "logs_root": "logs/"
        }
    except Exception as e:
        logger.error(f"Failed to load MIDI configuration: {e}")
        # Return minimal fallback configuration
        return {
            "pos_per_quarter": 12,
            "resolution": 480,
            "max_bar_length": 3,
            "max_bars": 512,
            "pad_idx": 338
        }, {
            "device": "cuda",
            "seed": 42,
            "debug": False,
            "output_root": "output/",
            "checkpoints_root": "checkpoints/",
            "logs_root": "logs/"
        }


# Load configurations with proper error handling
try:
    midi_config, global_config = get_midi_config_safe()
except Exception as e:
    logger.critical(f"Critical error loading MIDI configuration: {e}")
    # Set minimal fallback values to prevent import errors
    midi_config = {
        "pos_per_quarter": 12, 
        "resolution": 480, 
        "max_bar_length": 3,
        "max_bars": 512,
        "pad_idx": 338
    }
    global_config = {
        "device": "cuda",
        "seed": 42,
        "debug": False,
        "output_root": "output/",
        "checkpoints_root": "checkpoints/",
        "logs_root": "logs/"
    }

# parameters for input representation
DEFAULT_POS_PER_QUARTER = midi_config.get("pos_per_quarter", 12)
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32 + 1, dtype=int)
DEFAULT_DURATION_BINS = np.sort(
    np.concatenate(
        [
            np.arange(1, 13),  # smallest possible units up to 1 quarter
            np.arange(12, 24, 3)[1:],  # 16th notes up to 1 bar
            np.arange(13, 24, 4)[1:],  # triplets up to 1 bar
            np.arange(24, 48, 6),  # 8th notes up to 2 bars
            np.arange(48, 4 * 48, 12),  # quarter notes up to 8 bars
            np.arange(4 * 48, 16 * 48 + 1, 24),  # half notes up to 16 bars
        ]
    )
)
DEFAULT_TEMPO_BINS = np.linspace(0, 240, 32 + 1, dtype=int)
DEFAULT_NOTE_DENSITY_BINS = np.linspace(0, 12, 32 + 1)
DEFAULT_MEAN_VELOCITY_BINS = np.linspace(0, 128, 32 + 1)
DEFAULT_MEAN_PITCH_BINS = np.linspace(0, 128, 32 + 1)
DEFAULT_MEAN_DURATION_BINS = np.logspace(0, 7, 32 + 1, base=2)  # log space between 1 and 128 positions (~2.5 bars)

# parameters for output
DEFAULT_RESOLUTION = midi_config.get("resolution", 480)

# maximum length of a single bar is 3*4 = 12 beats
MAX_BAR_LENGTH = midi_config.get("max_bar_length", 3)
# maximum number of bars in a piece is 512 (this covers almost all sequences)
MAX_N_BARS = midi_config.get("max_bars", 512)
# padding token index
PAD_IDX = midi_config.get("pad_idx", 338)

# Log the loaded configuration for debugging
logger.info(f"Loaded MIDI constants - pos_per_quarter: {DEFAULT_POS_PER_QUARTER}, resolution: {DEFAULT_RESOLUTION}, max_bars: {MAX_N_BARS}")
