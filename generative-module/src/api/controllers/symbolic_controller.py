from fastapi import APIRouter

from application.symbolic.services.symbolic_service import (
    synthesize_midi as _synthesize_midi,
    encode_midi as _encode_midi,
    encode_dataset as _encode_dataset,
    tokenize_remi_dataset as _tokenize_remi_dataset,
)
from domain.models.symbolic.symbolic_model import EncodeParameters, EncodeDatasetParameters, TokenizeRemiDatasetParameters

router = APIRouter()


@router.post("/synthesize_midi")
def synthesize_midi(file: str, out_dir: str) -> dict:
    """
    Description:
    ------------
        Synthesize MIDI

    Parameters:
    -----------
        parameters: file: str, out_dir: str

    Returns:
    --------
    dict
        A dictionary

    """
    return _synthesize_midi(file, out_dir)


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