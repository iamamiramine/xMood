import asyncio

from fastapi import APIRouter

from application.encoder.services import encoder_service
from domain.models.encoder.encoder_model import EncodeParameters

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
def encode_dataset(config_path: str) -> dict:
    """
    Description:
    ------------
        Encode MIDI

    Parameters:
    -----------
        config: str

    Returns:
    --------
    dict
        A dictionary

    """
    # return encoder_service.encode_dataset(parameters)
    return asyncio.run(encoder_service.encode_dataset(config_path))
