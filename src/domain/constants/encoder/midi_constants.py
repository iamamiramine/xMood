import os
import json
import numpy as np

# Load config file
def load_midi_config():
    config_path = os.path.join("shared", "assets", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get('midi', {}), config.get('dataloader', {})

midi_config, datamodule_config = load_midi_config()

# parameters for input representation
DEFAULT_POS_PER_QUARTER = midi_config.get('pos_per_quarter', 12)
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
DEFAULT_RESOLUTION = midi_config.get('resolution', 480)

# maximum length of a single bar is 3*4 = 12 beats
MAX_BAR_LENGTH = midi_config.get('max_bar_length', 3)
# maximum number of bars in a piece is 512 (this covers almost all sequences)
MAX_N_BARS = datamodule_config.get('max_bars', 512)
