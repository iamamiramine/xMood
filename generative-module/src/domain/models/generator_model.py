from pydantic import BaseModel


class GenerateFromMIDIParameters(BaseModel):
    """Parameters for generating MIDI from existing MIDI files."""

    device: str
    latent_midi: str
    symbolic_midi: str
    emotions_midi: str
    output_folder: str
    output_name: str
    load_from_checkpoint: bool
    load_weights: bool
    checkpoint_path: str
    weights_path: str
    config_path: str
    context_size: int = 256
    max_bars: int = 16
    max_positions: int = 512
    max_n_tokens: int = 1024
    temperature: float = 0.8
    initial_context: int = 256
