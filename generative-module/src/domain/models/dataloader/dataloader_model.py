from fastapi import Query

from typing import Annotated, Tuple, Optional

from domain.models.base_model import BaseEnum
from pydantic import BaseModel, Field


class DataloaderParameters(BaseModel):
    dataset_name: Annotated[str, Query(description="Name of Dataset")]
    context_size: Annotated[int, Query(description="Number of tokens in the context to be passed to the auto-encoder")] = 256  # ALso max_len
    max_positions: Annotated[int, Query(description="Maximum position tokens in an input midi file")] = 2048
    max_bars: Annotated[int, Query(description="Maximum bars in an input midi file")] = 512
    max_bars_per_context: Annotated[int, Query(description="Add <bos> and <eos> to each context if contexts are limited to a certain number of bars")] = -1
    max_contexts_per_file: Annotated[int, Query(description="Maximum contexts in an input midi file after encoding")] = -1
    bar_token_mask: Annotated[str, Query(description="Bar masking token")] = None  # '<mask>'
    bar_token_idx: Annotated[int, Query(description="Bar token index")] = 2
    batch_size: Annotated[int, Query(description="Batch Size")] = 32
    num_workers: Annotated[int, Query(description="Number of workers to load the dataset")] = 2
    pin_memory: Annotated[bool, Query(description="whether to use pinned memory for loading data")] = True
    train_val_test_split: Annotated[Tuple[float, float, float], Query(description="split percentages for the dataset")] = (0.7, 0.2, 0.1)


class DataloaderModuleParameters(BaseModel):
    """Parameters for initializing DataloaderModule instances."""
    
    # Dataset configuration
    dataset_name: str = Field(..., description="Name of the dataset")
    
    # Context and sequence configuration
    context_size: int = Field(256, description="Number of tokens in the context")
    max_positions: int = Field(2048, description="Maximum position tokens in an input midi file")
    max_bars: int = Field(512, description="Maximum bars in an input midi file")
    max_bars_per_context: int = Field(-1, description="Add <bos> and <eos> to each context if contexts are limited to a certain number of bars")
    max_contexts_per_file: int = Field(-1, description="Maximum contexts in an input midi file after encoding")
    
    # Token configuration
    bar_token_mask: Optional[str] = Field(None, description="Bar masking token")
    bar_token_idx: int = Field(2, description="Bar token index")
    
    # Batch and processing configuration
    batch_size: int = Field(32, description="Batch size")
    num_workers: int = Field(2, description="Number of workers to load the dataset")
    pin_memory: bool = Field(True, description="Whether to use pinned memory for loading data")
    
    # Data split configuration
    train_val_test_split: Tuple[float, float, float] = Field((0.7, 0.2, 0.1), description="Split percentages for the dataset")
    
    # Data loading configuration
    load_latent: bool = Field(True, description="Whether to load latent representations")
    load_symb: bool = Field(True, description="Whether to load symbolic features")
    load_emotions: bool = Field(True, description="Whether to load emotion data")
    load_global_features: bool = Field(False, description="Whether to load global features")
    load_text_prompts: bool = Field(False, description="Whether to load text prompts")
    
    # Processing modes
    encode: bool = Field(False, description="Whether to encode data")
    caption: bool = Field(False, description="Whether to generate captions")


class DatasetLoadParameters(BaseModel):
    """Parameters for loading and processing datasets."""
    
    # Dataset configuration
    dataset_name: str = Field(..., description="Name of the dataset")
    stage: str = Field("fit", description="Training stage (fit, test, predict)")
    
    # Data loading configuration
    load_latent: bool = Field(True, description="Whether to load latent representations")
    load_symb: bool = Field(True, description="Whether to load symbolic features")
    load_emotions: bool = Field(True, description="Whether to load emotion data")
    load_global_features: bool = Field(False, description="Whether to load global features")
    load_text_prompts: bool = Field(False, description="Whether to load text prompts")
    
    # Processing configuration
    batch_size: int = Field(32, description="Batch size")
    num_workers: int = Field(2, description="Number of workers")
    pin_memory: bool = Field(True, description="Whether to use pinned memory")
    
    # Output configuration
    output_dir: Optional[str] = Field(None, description="Output directory for processed data")
    save_processed: bool = Field(False, description="Whether to save processed data")
