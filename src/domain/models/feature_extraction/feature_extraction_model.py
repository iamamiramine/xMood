from fastapi import Query
from typing import Annotated, Optional, Union
from pydantic import BaseModel
import pretty_midi as pm

from domain.models.base_model import BaseEnum


class SymbolicFeaturesParameters(BaseModel):
    midi: str
    processed_dir: Annotated[
        Optional[str], Query(description="Directory containing the processed pkl file. If not provided, will use PROCESSED_PATH/dataset_name")
    ] = None
    save: Annotated[bool, Query(description="Whether to save the output")] = True
