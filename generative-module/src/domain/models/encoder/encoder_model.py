import pretty_midi
from fastapi import Query

from typing import Annotated, Tuple, Optional

from pydantic import BaseModel, Field

# Import constants
from domain.constants.model_constants import ModelConstants
from domain.constants.encoder.midi_constants import MAX_N_BARS, DEFAULT_POS_PER_QUARTER

# Import base parameter classes
from domain.models.base_parameters import EncoderBaseParameters


class EncodeParameters(BaseModel):
    """Parameters for single MIDI encoding - lightweight API model"""
    
    midi: Annotated[str, Query(description="Input MIDI file")]
    save: Annotated[bool, Query(description="Save the output file")] = False
    encodings_out_dir: Annotated[str, Query(description="Output directory for the encoded MIDI file")] = None
    dataset_key: Annotated[str | None, Query(description="Key from the dataset (e.g. 'F# minor')")] = None


class EncodeDatasetParameters(EncoderBaseParameters):
    """Parameters for encoding an entire dataset - inherits from base"""
    
    # Additional parameters specific to dataset encoding
    max_files: Optional[int] = Field(None, description="Maximum number of files to process")
    resume_from: Optional[str] = Field(None, description="Resume processing from specific file")
    
    # Output configuration
    save: bool = Field(True, description="Save the encoded output")
    encodings_out_dir: Optional[str] = Field(None, description="Output directory for encoded files")
    overwrite_existing: bool = Field(False, description="Overwrite existing encoded files")
    
    # Validation configuration
    validate_output: bool = Field(True, description="Validate encoded output")
    skip_invalid: bool = Field(True, description="Skip invalid MIDI files")


class TokenizeRemiDatasetParameters(EncoderBaseParameters):
    """Parameters for tokenizing REMI dataset - inherits from base"""
    
    # Additional parameters specific to tokenization
    max_files: Optional[int] = Field(None, description="Maximum number of files to process")
    resume_from: Optional[str] = Field(None, description="Resume processing from specific file")
    
    # Output configuration
    save: bool = Field(True, description="Save tokenized output")
    tokens_out_dir: Optional[str] = Field(None, description="Output directory for tokenized files")
    overwrite_existing: bool = Field(False, description="Overwrite existing tokenized files")
    
    # Vocabulary configuration
    vocab_path: Optional[str] = Field(None, description="Path to vocabulary file")
    create_vocab: bool = Field(True, description="Create vocabulary if not exists")
    
    # Validation configuration
    validate_output: bool = Field(True, description="Validate tokenized output")
    skip_invalid: bool = Field(True, description="Skip invalid files")
