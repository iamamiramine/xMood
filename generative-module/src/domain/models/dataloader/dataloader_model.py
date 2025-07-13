from fastapi import Query

from typing import Annotated, Tuple, Optional

from pydantic import BaseModel, Field

# Import constants
from domain.constants.model_constants import ModelConstants
from domain.constants.midi_constants import MAX_N_BARS


class DataloaderParameters(BaseModel):
    dataset_name: Annotated[str, Query(description="Name of Dataset")] = "DATASET_NAME"
    context_size: Annotated[int, Query(description="Number of tokens in the context to be passed to the auto-symbolic")] = ModelConstants.DEFAULT_CONTEXT_SIZE
    max_positions: Annotated[int, Query(description="Maximum position tokens in an input midi file")] = ModelConstants.DEFAULT_MAX_POSITIONS
    max_bars: Annotated[int, Query(description="Maximum bars in an input midi file")] = MAX_N_BARS
    max_bars_per_context: Annotated[int, Query(description="Add <bos> and <eos> to each context if contexts are limited to a certain number of bars")] = -1
    max_contexts_per_file: Annotated[int, Query(description="Maximum contexts in an input midi file after encoding")] = -1
    bar_token_mask: Annotated[Optional[str], Query(description="Bar masking token")] = None
    bar_token_idx: Annotated[int, Query(description="Bar token index")] = ModelConstants.BOS_TOKEN_ID
    batch_size: Annotated[int, Query(description="Batch Size")] = ModelConstants.DEFAULT_BATCH_SIZE
    num_workers: Annotated[int, Query(description="Number of workers to load the dataset")] = 2
    pin_memory: Annotated[bool, Query(description="whether to use pinned memory for loading data")] = True
    train_val_test_split: Annotated[Tuple[float, float, float], Query(description="split percentages for the dataset")] = (0.7, 0.2, 0.1)


class DataloaderModuleParameters(BaseModel):
    """Parameters for initializing DataloaderModule instances."""
    
    # Dataset configuration
    dataset_name: str = Field("DATASET_NAME", description="Name of the dataset")
    
    # Context and sequence configuration
    context_size: int = Field(ModelConstants.DEFAULT_CONTEXT_SIZE, description="Number of tokens in the context")
    max_positions: int = Field(ModelConstants.DEFAULT_MAX_POSITIONS, description="Maximum position tokens in an input midi file")
    max_bars: int = Field(MAX_N_BARS, description="Maximum bars in an input midi file")
    max_bars_per_context: int = Field(-1, description="Add <bos> and <eos> to each context if contexts are limited to a certain number of bars")
    max_contexts_per_file: int = Field(-1, description="Maximum contexts in an input midi file after encoding")
    
    # Token configuration
    bar_token_mask: Optional[str] = Field(None, description="Bar masking token")
    bar_token_idx: int = Field(ModelConstants.BOS_TOKEN_ID, description="Bar token index")
    
    # Batch and processing configuration
    batch_size: int = Field(ModelConstants.DEFAULT_BATCH_SIZE, description="Batch size")
    num_workers: int = Field(2, description="Number of workers to load the dataset")
    pin_memory: bool = Field(True, description="Whether to use pinned memory for loading data")
    
    # Data split configuration
    train_val_test_split: Tuple[float, float, float] = Field((0.7, 0.2, 0.1), description="Split percentages for the dataset")


class DatasetLoadParameters(BaseModel):
    """Parameters for loading and processing datasets."""
    
    # Dataset configuration
    dataset_name: str = Field("DATASET_NAME", description="Name of the dataset")
    stage: str = Field("fit", description="Training stage (fit, test, predict)")
    
    # Processing configuration
    batch_size: int = Field(ModelConstants.DEFAULT_BATCH_SIZE, description="Batch size")
    num_workers: int = Field(2, description="Number of workers")
    pin_memory: bool = Field(True, description="Whether to use pinned memory")
    
    # Output configuration
    output_dir: Optional[str] = Field(None, description="Output directory for processed data")
    save_processed: bool = Field(False, description="Whether to save processed data")
