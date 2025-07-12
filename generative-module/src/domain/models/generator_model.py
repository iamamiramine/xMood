from pydantic import BaseModel, Field
from typing import Optional

# Import constants
from domain.constants.model_constants import ModelConstants
from domain.constants.encoder.midi_constants import MAX_N_BARS, DEFAULT_POS_PER_QUARTER

# Import base parameter classes
from domain.models.base_parameters import GeneratorBaseParameters


class GenerateFromMIDIParameters(BaseModel):
    """Parameters for generating MIDI from existing MIDI files - lightweight API model"""

    device: str
    latent_midi: str
    symbolic_midi: str
    emotions_midi: str
    output_folder: str
    output_name: str
    load_from_checkpoint: bool
    load_weights: bool
    checkpoint_path: str
    weights_path: str
    config_path: str
    context_size: int = ModelConstants.DEFAULT_CONTEXT_SIZE
    max_bars: int = 16
    max_positions: int = ModelConstants.DEFAULT_MAX_POSITIONS
    max_n_tokens: int = ModelConstants.DEFAULT_MAX_N_TOKENS
    temperature: float = ModelConstants.DEFAULT_TEMPERATURE
    initial_context: int = ModelConstants.DEFAULT_CONTEXT_SIZE


class GeneratorTrainingParameters(GeneratorBaseParameters):
    """Parameters for training generator models - inherits from base"""
    
    # Additional generator-specific parameters
    max_n_tokens: int = Field(ModelConstants.DEFAULT_MAX_N_TOKENS, description="Maximum number of tokens")
    temperature: float = Field(ModelConstants.DEFAULT_TEMPERATURE, description="Temperature for sampling")
    initial_context: int = Field(ModelConstants.DEFAULT_CONTEXT_SIZE, description="Initial context size")
