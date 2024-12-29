from fastapi import Query

from typing import Annotated, Tuple, Optional

from src.domain.models.base_model import BaseEnum
from pydantic import BaseModel


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
