from fastapi import Query

from typing import Annotated, Optional

from pydantic import BaseModel, Field

# Import constants
from domain.constants.model_constants import ModelConstants
from domain.constants.midi_constants import MAX_N_BARS


class MusicBaseParameters(BaseModel):
    midi: Annotated[str, Query(description="Input MIDI file")]
    save: Annotated[bool, Query(description="Save the output file")] = False
    out_dir: Annotated[str, Query(description="Output directory for the encoded MIDI file")] = None


class EncodeParameters(MusicBaseParameters):
    processed_dir: Annotated[
        Optional[str],
        Query(
            description="Directory containing the processed pkl file. If not provided, will use PROCESSED_PATH/dataset_name"
        ),
    ] = None
    add_position_tokens: bool = Field(
        False, description="Whether to add position tokens before features (only applies to bar-level features)"
    )
    omit_time_sig: bool = Field(False, description="Whether to omit time signature features")
    omit_instruments: bool = Field(False, description="Whether to omit instrument features")
    omit_chords: bool = Field(False, description="Whether to omit chord features")
    omit_meta: bool = Field(False, description="Whether to omit meta features")


class EncodeDatasetParameters(EncodeParameters):
    """Parameters for encoding an entire dataset."""

    # Dataset configuration
    dataset_name: str = Field("DATASET_NAME", description="Name of the dataset")

    # Processing configuration
    batch_size: int = Field(ModelConstants.DEFAULT_BATCH_SIZE, description="Batch size for processing")
    num_workers: int = Field(4, description="Number of workers for parallel processing")
    max_files: Optional[int] = Field(None, description="Maximum number of files to process")
    resume_from: Optional[str] = Field(None, description="Resume processing from specific file")

    # Output configuration
    save: bool = Field(True, description="Save the encoded output")
    encodings_out_dir: Optional[str] = Field(None, description="Output directory for encoded files")
    overwrite_existing: bool = Field(False, description="Overwrite existing encoded files")

    # Device configuration
    device: str = Field("cuda", description="Device to use for processing")

    # Validation configuration
    validate_output: bool = Field(True, description="Validate encoded output")
    skip_invalid: bool = Field(True, description="Skip invalid MIDI files")


class TokenizeRemiDatasetParameters(BaseModel):
    """Parameters for tokenizing REMI dataset."""

    # Dataset configuration
    dataset_name: str = Field("DATASET_NAME", description="Name of the dataset")

    # Context and sequence configuration
    context_size: int = Field(ModelConstants.DEFAULT_CONTEXT_SIZE, description="Context size for tokenization")
    max_positions: int = Field(ModelConstants.DEFAULT_MAX_POSITIONS, description="Maximum position tokens")
    max_bars: int = Field(MAX_N_BARS, description="Maximum bars")
    max_bars_per_context: int = Field(-1, description="Maximum bars per context")
    max_contexts_per_file: int = Field(-1, description="Maximum contexts per file")

    # Token configuration
    bar_token_mask: Optional[str] = Field(None, description="Bar masking token")
    bar_token_idx: int = Field(ModelConstants.BOS_TOKEN_ID, description="Bar token index")

    # Processing configuration
    batch_size: int = Field(ModelConstants.DEFAULT_BATCH_SIZE, description="Batch size for processing")
    num_workers: int = Field(4, description="Number of workers")
    max_files: Optional[int] = Field(None, description="Maximum number of files to process")
    resume_from: Optional[str] = Field(None, description="Resume processing from specific file")

    # Output configuration
    save: bool = Field(True, description="Save tokenized output")
    tokens_out_dir: Optional[str] = Field(None, description="Output directory for tokenized files")
    overwrite_existing: bool = Field(False, description="Overwrite existing tokenized files")

    # Vocabulary configuration
    vocab_path: Optional[str] = Field(None, description="Path to vocabulary file")
    create_vocab: bool = Field(True, description="Create vocabulary if not exists")

    # Device configuration
    device: str = Field("cuda", description="Device to use for processing")

    # Validation configuration
    validate_output: bool = Field(True, description="Validate tokenized output")
    skip_invalid: bool = Field(True, description="Skip invalid files")
