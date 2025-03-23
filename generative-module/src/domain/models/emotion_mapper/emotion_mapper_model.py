from dataclasses import dataclass
from typing import Optional


@dataclass
class EmotionMapperTrainingParameters:
    dataset_name: str
    training_name: str
    device: str = "cuda"
    batch_size: int = 32
    epochs: int = 100
    max_training_steps: Optional[int] = None
    feature_dim: int = 512
    d_model: int = 512
    num_heads: int = 8
    max_bars: int = 512
    load_from_checkpoint: bool = False
    checkpoint_name: Optional[str] = None


@dataclass
class EmotionMapperGenerateParameters:
    dataset_name: str
    training_name: str
    checkpoint_name: str
    device: str = "cuda"
    max_n_files: int = -1
    verbose: bool = False
