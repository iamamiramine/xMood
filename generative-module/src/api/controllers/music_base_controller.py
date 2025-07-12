from typing import Any

from fastapi import APIRouter

from domain.models.music_base.music_base_model import (
    MusicBaseParameters,
)
from application.music_base.services import music_base_service

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
    return music_base_service.synthesize_midi(file, out_dir)


@router.post("/extract_chords")
def extract_chords(parameters: MusicBaseParameters) -> Any:
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
    return music_base_service.extract_chords(parameters)

