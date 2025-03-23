from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union


@dataclass
class TokenizerParameters:
    """Parameters for creating and saving a tokenizer"""
    vocab_size: int = 50000
    special_tokens: List[str] = field(default_factory=list)
    output_dir: Path = Path("./tokenizer")
    tokenizer_filename: str = "music_tokenizer.json"
    use_bpe: bool = True
    use_existing: bool = False
    existing_tokenizer_path: Optional[Path] = None


@dataclass
class DatasetParameters:
    """Parameters for creating text datasets from MIDI data"""
    midi_dir: Path
    output_dir: Path
    processed_dir: Path
    split_ratio: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    max_samples: Optional[int] = None
    include_metadata: bool = True
    add_moods: bool = True
    add_features: bool = True
    include_chord_progression: bool = True
    include_key_signature: bool = True
    tokenize: bool = True
    max_length: int = 1024
    create_projection_data: bool = True
    create_generation_data: bool = True
    features_output_dir: Optional[Path] = None


@dataclass
class ProjectionModelParameters:
    """Parameters for the projection model in the integrated pipeline"""
    d_model: int = 512
    encoder_layers: int = 6
    decoder_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    checkpoint_path: Optional[Path] = None
    mood_vocab_size: int = 100
    feature_embedding_size: int = 1024
    context_size: int = 256
    max_bars: int = 512
    max_positions: int = 1024
    output_dir: Path = Path("./projection_model")


@dataclass
class ModelExportParameters:
    """Parameters for exporting a model to text-generation-webui format"""
    model_type: str = "t5"  # t5, bart, custom
    model_name: str = "music_generation_model"
    source_checkpoint: Optional[Path] = None
    projection_checkpoint: Optional[Path] = None
    generator_checkpoint: Optional[Path] = None
    tokenizer_path: Path
    config_overrides: Dict = field(default_factory=dict)
    d_model: int = 512
    num_encoder_layers: int = 6
    num_decoder_layers: int = 12
    num_heads: int = 8
    d_ff: int = 2048
    output_dir: Path = Path("./exported_model")
    webui_dir: Optional[Path] = None
    convert_format: bool = True
    use_integrated_pipeline: bool = False


@dataclass
class MigrationConfig:
    """Overall configuration for the migration process"""
    tokenizer: TokenizerParameters
    dataset: DatasetParameters
    model_export: ModelExportParameters
    projection: Optional[ProjectionModelParameters] = None
    create_extension: bool = True
    extension_name: str = "music_generator"
    convert_existing_checkpoints: bool = False
    existing_checkpoint_path: Optional[Path] = None
    integrated_pipeline: bool = False 