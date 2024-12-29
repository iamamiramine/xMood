import asyncio

from fastapi import APIRouter

from src.application.encoder.services import encoder_service
from src.domain.models.encoder.encoder_model import (
    EncodeDatasetParameters,
    EncodeParameters,
)

router = APIRouter()


@router.post("/encode_midi")
def encode_midi(parameters: EncodeParameters) -> dict:
    """
    Description:
    ------------
        Encode MIDI

    Parameters:
    -----------
        parameters: EncodeMidiParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return encoder_service.encode_midi(parameters)


@router.post("/encode_dataset")
def encode_dataset(parameters: EncodeDatasetParameters) -> dict:
    """
    Description:
    ------------
        Encode MIDI

    Parameters:
    -----------
        dataset_name: str = ""

    Returns:
    --------
    dict
        A dictionary

    """
    # return encoder_service.encode_dataset(parameters)
    return asyncio.run(encoder_service.encode_dataset(parameters))
