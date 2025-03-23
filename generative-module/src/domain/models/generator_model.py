from pydantic import BaseModel


class GenerateFromMIDIParameters(BaseModel):
    """Parameters for generating MIDI from existing MIDI files."""

    latent_midi: str
    symbolic_midi: str
    emotions_midi: str
    output_folder: str
    output_name: str
    checkpoint_path: str
    context_size: int = 256
    max_bars: int = 16
    max_positions: int = 512
    max_n_tokens: int = 1024
    temperature: float = 0.8
