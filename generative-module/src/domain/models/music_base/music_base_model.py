from fastapi import Query

from typing import Annotated, Tuple, Optional

from domain.models.base_model import BaseEnum
from pydantic import BaseModel


class MusicBaseParameters(BaseModel):
    midi: Annotated[str, Query(description="Input MIDI file")]
    save: Annotated[bool, Query(description="Save the output file")] = False
    out_dir: Annotated[str, Query(description="Output directory for the encoded MIDI file")] = None
    api_call: Annotated[bool, Query(description="API Call Bool")] = False
