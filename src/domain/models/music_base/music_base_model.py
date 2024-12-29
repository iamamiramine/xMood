from fastapi import Query

from typing import Annotated, Tuple, Optional

from src.domain.models.base_model import BaseEnum
from pydantic import BaseModel


class MusicBaseParameters(BaseModel):
    midi: Annotated[str, Query(description="Input MIDI file")]
    save: Annotated[bool, Query(description="Save the output file")] = False
    out_dir: Annotated[str, Query(description="Output directory for the encoded MIDI file")] = None
    api_call: Annotated[bool, Query(description="API Call Bool")] = False


class TonalPlanParameters(MusicBaseParameters):
    alpha: Annotated[float, Query(description="Alpha value for the tonal plan")] = 1.0
    beta: Annotated[float, Query(description="Beta value for the tonal plan")] = 1.0
    gamma: Annotated[float, Query(description="Gamma value for the tonal plan")] = 1.0
    c: Annotated[float, Query(description="C value for the tonal plan")] = 1.0
    w: Annotated[float, Query(description="W value for the tonal plan")] = 1.0
    chords: Annotated[dict | str, Query(description="Chords extracted from the MIDI file")] = None
