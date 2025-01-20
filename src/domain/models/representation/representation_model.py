from fastapi import Query

from typing import Annotated, Tuple, Optional

from src.domain.models.base_model import BaseEnum
from pydantic import BaseModel


class RepresentationParameters(BaseModel):
    midi: Annotated[str, Query(description="Input MIDI file")]
    save: Annotated[bool, Query(description="Save the output file")] = False
    out_dir: Annotated[str, Query(description="Output directory for the encoded MIDI file")] = None
    api_call: Annotated[bool, Query(description="API Call Bool")] = False

    dataset_name: Annotated[str, Query(description="Dataset Name")]
    context_size: Annotated[int, Query(description="Context Size")] = 256
    max_bars: Annotated[int, Query(description="Max bars")] = 512
    max_positions: Annotated[int, Query(description="Max positions")] = 1024
    bar_token_mask: Annotated[str, Query(description="Bar token mask")] = None
    max_bars_per_context: Annotated[int, Query(description="Max Bars per Context")] = -1
    max_contexts_per_file: Annotated[int, Query(description="Max contexts per file")] = -1
    load_symb: Annotated[bool, Query(description="Load Symbolic Features Features")] = False
    load_latent: Annotated[bool, Query(description="Load Latent Features")] = False
    load_emotions: Annotated[bool, Query(description="Load Emotion Features")] = False
