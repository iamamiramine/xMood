import asyncio

from fastapi import APIRouter

from application.encoder.services.encoder_service import (
    encode_midi as _encode_midi,
    encode_dataset as _encode_dataset,
    tokenize_remi_dataset as _tokenize_remi_dataset,
)
from domain.models.encoder.encoder_model import EncodeParameters, EncodeDatasetParameters, TokenizeRemiDatasetParameters

router = APIRouter()


@router.post("/encode_midi")
def encode_midi(parameters: EncodeParameters) -> dict:
    """
    Encode a single MIDI file using REMI representation.
    
    Args:
        parameters: EncodeParameters containing MIDI file path and processing options
        
    Returns:
        dict: Encoding result message
    """
    try:
        result = _encode_midi(parameters)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/encode_dataset")
async def encode_dataset(parameters: EncodeDatasetParameters) -> dict:
    """
    Encode an entire dataset of MIDI files using REMI representation.
    
    Args:
        parameters: EncodeDatasetParameters containing dataset configuration
        
    Returns:
        dict: Encoding result summary
    """
    try:
        result = await _encode_dataset(parameters)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/tokenize_remi_dataset")
async def tokenize_remi_dataset(parameters: TokenizeRemiDatasetParameters) -> dict:
    """
    Tokenize REMI sequences from an encoded dataset.
    
    Args:
        parameters: TokenizeRemiDatasetParameters containing tokenization configuration
        
    Returns:
        dict: Tokenization result summary
    """
    try:
        result = await _tokenize_remi_dataset(parameters)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}