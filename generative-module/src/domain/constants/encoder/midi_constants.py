import os
import json
import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def load_midi_config() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load MIDI configuration from config file with proper error handling.
    
    Returns:
        Tuple of (midi_config, datamodule_config) dictionaries
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        json.JSONDecodeError: If the configuration file contains invalid JSON
        Exception: For other configuration loading errors
    """
    config_path = os.path.join("shared", "config", "config.json")
    
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        if not isinstance(config, dict):
            raise ValueError(f"Configuration file must contain a JSON object, got {type(config)}")
        
        midi_config = config.get("midi", {})
        datamodule_config = config.get("dataloader", {})
        
        logger.info(f"Successfully loaded MIDI configuration from {config_path}")
        return midi_config, datamodule_config
        
    except FileNotFoundError as e:
        logger.error(f"MIDI configuration file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in MIDI configuration file at {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading MIDI configuration from {config_path}: {e}")
        raise


def get_midi_config_safe() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load MIDI configuration with fallback values.
    
    Returns:
        Tuple of (midi_config, datamodule_config) dictionaries with fallback values
    """
    try:
        return load_midi_config()
    except FileNotFoundError:
        logger.warning("MIDI configuration file not found, using default values")
        return {
            "pos_per_quarter": 12,
            "resolution": 480,
            "max_bar_length": 3,
        }, {
            "max_bars": 512,
        }
    except Exception as e:
        logger.error(f"Failed to load MIDI configuration: {e}")
        # Return minimal fallback configuration
        return {
            "pos_per_quarter": 12,
            "resolution": 480,
            "max_bar_length": 3,
        }, {
            "max_bars": 512,
        }


# Load configurations with proper error handling
try:
    midi_config, datamodule_config = get_midi_config_safe()
except Exception as e:
    logger.critical(f"Critical error loading MIDI configuration: {e}")
    # Set minimal fallback values to prevent import errors
    midi_config = {"pos_per_quarter": 12, "resolution": 480, "max_bar_length": 3}
    datamodule_config = {"max_bars": 512}

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
MAX_N_BARS = datamodule_config.get("max_bars", 512)
