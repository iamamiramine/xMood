import pretty_midi
from fastapi import Query

from typing import Annotated, Tuple, Optional

from pydantic import BaseModel, Field


class EncodeParameters(BaseModel):
    midi: Annotated[str, Query(description="Input MIDI file")]
    save: Annotated[bool, Query(description="Save the output file")] = False
    encodings_out_dir: Annotated[str, Query(description="Output directory for the encoded MIDI file")] = None
    dataset_key: Annotated[str | None, Query(description="Key from the dataset (e.g. 'F# minor')")] = None


class EncodeDatasetParameters(BaseModel):
    """Parameters for encoding an entire dataset."""
    
    # Dataset configuration
    dataset_name: str = Field("ReMIDICaps", description="Name of the dataset")
    
    # Processing configuration
    batch_size: int = Field(32, description="Batch size for processing")
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
    dataset_name: str = Field("ReMIDICaps", description="Name of the dataset")
    
    # Context and sequence configuration
    context_size: int = Field(256, description="Context size for tokenization")
    max_positions: int = Field(2048, description="Maximum position tokens")
    max_bars: int = Field(512, description="Maximum bars")
    max_bars_per_context: int = Field(-1, description="Maximum bars per context")
    max_contexts_per_file: int = Field(-1, description="Maximum contexts per file")
    
    # Token configuration
    bar_token_mask: Optional[str] = Field(None, description="Bar masking token")
    bar_token_idx: int = Field(2, description="Bar token index")
    
    # Processing configuration
    batch_size: int = Field(32, description="Batch size for processing")
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
