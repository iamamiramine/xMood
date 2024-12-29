import pretty_midi
from fastapi import Query

from typing import Annotated, Tuple, Optional

from pydantic import BaseModel


class EncodeParameters(BaseModel):
    midi: Annotated[str, Query(description="Input MIDI file")]
    alpha: Annotated[float, Query(description="Alpha value for the tonal plan")] = 1.0
    beta: Annotated[float, Query(description="Beta value for the tonal plan")] = 1.0
    gamma: Annotated[float, Query(description="Gamma value for the tonal plan")] = 1.0
    c: Annotated[float, Query(description="C value for the tonal plan")] = 1.0
    w: Annotated[float, Query(description="W value for the tonal plan")] = 1.0
    save: Annotated[bool, Query(description="Save the output file")] = False
    encodings_out_dir: Annotated[str, Query(description="Output directory for the encoded MIDI file")] = None
    chords_out_dir: Annotated[str, Query(description="Output directory for the encoded MIDI file")] = None
    keys_out_dir: Annotated[str, Query(description="Output directory for the encoded MIDI file")] = None


class EncodeDatasetParameters(BaseModel):
    dataset_name: Annotated[str, Query(description="Name of Dataset")]
    alpha: Annotated[float, Query(description="Alpha value for the tonal plan")] = 1.0
    beta: Annotated[float, Query(description="Beta value for the tonal plan")] = 1.0
    gamma: Annotated[float, Query(description="Gamma value for the tonal plan")] = 1.0
    c: Annotated[float, Query(description="C value for the tonal plan")] = 1.0
    w: Annotated[float, Query(description="W value for the tonal plan")] = 1.0
    save: Annotated[bool, Query(description="Save the output file")] = False
