from typing import Any

from fastapi import APIRouter

from src.domain.models.music_base.music_base_model import (
    TonalPlanParameters,
    MusicBaseParameters,
)
from src.application.music_base.services import music_base_service

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


@router.post("/estimate_tonal_plan")
def estimate_tonal_plan(parameters: TonalPlanParameters) -> Any:
    """
    Description:
    ------------
        Encode MIDI

    Parameters:
    -----------
        chords: str = "path/to/chords.pkl",

    Returns:
    --------
    dict
        A dictionary

    """
    return music_base_service.estimate_tonal_plan(parameters)
