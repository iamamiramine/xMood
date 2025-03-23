from fastapi import APIRouter

from application.generator.services import generator_service
from domain.models.generator_model import GenerateFromMIDIParameters

router = APIRouter()


@router.post("/train_generator")
def train_generator(config_path: str) -> dict:
    """
    Description:
    ------------
        Train Generator

    Parameters:
    -----------
        config_path: str

    Returns:
    --------
    dict
        A dictionary

    """
    return generator_service.train_generator(config_path)


@router.post("/generate_from_midi")
def generate_from_midi(parameters: GenerateFromMIDIParameters) -> dict:
    """
    Description:
    ------------
        Generate a new MIDI file using an existing MIDI file as prompt

    Parameters:
    -----------
        parameters: GenerateFromMIDIParameters
            Parameters for MIDI generation including:
            - midi_path: Path to the source MIDI file
            - output_folder: Folder name for the generated output
            - output_name: Name for the generated file
            - checkpoint_path: Path to the generator checkpoint
            - context_size: Size of the context window (default: 256)
            - max_bars: Maximum number of bars to generate (default: 16)
            - max_positions: Maximum number of positions (default: 512)
            - max_n_tokens: Maximum number of tokens to generate (default: 1024)
            - temperature: Sampling temperature (default: 0.8)

    Returns:
    --------
    dict
        A dictionary containing the generation status and output path
    """
    return generator_service.generate_from_midi(parameters)
