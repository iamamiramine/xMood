from fastapi import Query

from typing import Annotated, Tuple, Optional

from domain.models.base_model import BaseEnum
from pydantic import BaseModel, Field

# Import constants
from domain.constants.model_constants import ModelConstants
from domain.constants.encoder.midi_constants import MAX_N_BARS, DEFAULT_POS_PER_QUARTER

# Import base parameter classes
from domain.models.base_parameters import DataloaderBaseParameters


class DataloaderParameters(DataloaderBaseParameters):
    """Parameters for dataloader processing - legacy API compatibility"""
    
    # Legacy API parameters with Query annotations for FastAPI compatibility
    dataset_name: Annotated[str, Query(description="Name of Dataset")] = "DATASET_NAME"
    context_size: Annotated[int, Query(description="Number of tokens in the context to be passed to the auto-encoder")] = ModelConstants.DEFAULT_CONTEXT_SIZE
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


class DataloaderModuleParameters(DataloaderBaseParameters):
    """Parameters for initializing DataloaderModule instances - inherits from base"""
    pass


class DatasetLoadParameters(DataloaderBaseParameters):
    """Parameters for loading and processing datasets - inherits from base"""
    
    # Additional parameters specific to dataset loading
    stage: str = Field("fit", description="Training stage (fit, test, predict)")
    output_dir: Optional[str] = Field(None, description="Output directory for processed data")
    save_processed: bool = Field(False, description="Whether to save processed data")
