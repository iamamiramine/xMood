"""
Constants for classifier-related functionality.
"""
import os
import numpy as np


class ClassifierConstants:
    """Constants for classifier service."""
    
    # Dataset paths
    EMOPIA_DATA_ROOT = "datasets/EMOPIA/"
    DICTIONARY_FILENAME = "dictionary.pkl"
    
    # Model configuration
    DEFAULT_VOCAB_SIZE = 1382
    DEFAULT_LSTM_HIDDEN_DIM = 128
    
    # Model paths
    OUTPUT_BASE_PATH = "output"
    DEMOS_PATH = "demos"
    DEMO_2_PATH = "demo_2"
    CLASSIFIER_PATH = "classifier"
    REMIDI_CAPS_PATH = "ReMIDICaps"
    HPARAMS_FILENAME = "hparams.yaml"
    CHECKPOINT_FILENAME = "last-v1.ckpt"
    
    @classmethod
    def get_dictionary_path(cls) -> str:
        """Get the full path to the dictionary file."""
        return os.path.join(cls.EMOPIA_DATA_ROOT, cls.DICTIONARY_FILENAME)
    
    @classmethod
    def get_model_config_path(cls, task: str, model_type: str) -> str:
        """Get the full path to the model configuration file."""
        return os.path.join(
            cls.OUTPUT_BASE_PATH, 
            cls.DEMOS_PATH, 
            cls.DEMO_2_PATH,
            cls.CLASSIFIER_PATH,
            cls.REMIDI_CAPS_PATH,
            task,
            model_type,
            cls.HPARAMS_FILENAME
        )
    
    @classmethod
    def get_model_checkpoint_path(cls, task: str, model_type: str) -> str:
        """Get the full path to the model checkpoint file."""
        return os.path.join(
            cls.OUTPUT_BASE_PATH,
            cls.DEMOS_PATH,
            cls.DEMO_2_PATH,
            cls.CLASSIFIER_PATH,
            cls.REMIDI_CAPS_PATH,
            task,
            model_type,
            cls.CHECKPOINT_FILENAME
        )


class MidiProcessingConstants:
    """Constants for MIDI processing."""
    
    # MIDI resolution settings
    BEAT_RESOL = 480
    BAR_RESOL = BEAT_RESOL * 4
    TICK_RESOL = BEAT_RESOL // 4
    
    # Instrument mapping
    INSTR_NAME_MAP = {"piano": 0}
    
    # Processing limits
    MIN_BPM = 40
    MIN_VELOCITY = 40
    
    # Note sorting: 0 = ascending, 1 = descending
    NOTE_SORTING = 1
    
    # Default binning
    DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 64 + 1, dtype=np.int32)
    DEFAULT_BPM_BINS = np.linspace(32, 224, 64 + 1, dtype=np.int32)
    DEFAULT_SHIFT_BINS = np.linspace(-60, 60, 60 + 1, dtype=np.int32)
    
    @classmethod
    def get_default_duration_bins(cls) -> np.ndarray:
        """Get default duration bins array."""
        return np.arange(cls.BEAT_RESOL / 8, cls.BEAT_RESOL * 8 + 1, cls.BEAT_RESOL / 8)
    
    # Pitch to note name mapping
    NUM2PITCH = {
        0: "C",
        1: "C#",
        2: "D",
        3: "D#",
        4: "E",
        5: "F",
        6: "F#",
        7: "G",
        8: "G#",
        9: "A",
        10: "A#",
        11: "B",
    }


class MoodMappingConstants:
    """Constants for mood category mapping between different formats."""
    
    # Mood categories used in pseudo labeller (image classification)
    CLIP_MOOD_CATEGORIES = [
        'relaxing', 'christmas', 'dramatic', 'meditative',
        'energetic', 'happy', 'motivational', 'dark', 'love'
    ]
    
    # Mapping from CLIP mood categories to ModelConstants.MOOD_NAMES
    CLIP_TO_MOOD_NAMES = {
        'relaxing': 'RELAXING_KEY',
        'christmas': 'CHRISTMAS_KEY',
        'dramatic': 'DRAMATIC_KEY',
        'meditative': 'MEDITATIVE_KEY',
        'energetic': 'ENERGETIC_KEY',
        'happy': 'HAPPY_KEY',
        'motivational': 'MOTIVATIONAL_KEY',
        'dark': 'DARK_KEY',
        'love': 'LOVE_KEY'
    }
    
    # Reverse mapping from ModelConstants.MOOD_NAMES to CLIP mood categories
    MOOD_NAMES_TO_CLIP = {v: k for k, v in CLIP_TO_MOOD_NAMES.items()}
    
    @classmethod
    def get_clip_mood_categories(cls) -> list:
        """Get the CLIP mood categories list."""
        return cls.CLIP_MOOD_CATEGORIES
    
    @classmethod
    def clip_to_mood_name(cls, clip_mood: str) -> str:
        """Convert CLIP mood category to ModelConstants mood name."""
        return cls.CLIP_TO_MOOD_NAMES.get(clip_mood, clip_mood)
    
    @classmethod
    def mood_name_to_clip(cls, mood_name: str) -> str:
        """Convert ModelConstants mood name to CLIP mood category."""
        return cls.MOOD_NAMES_TO_CLIP.get(mood_name, mood_name) 