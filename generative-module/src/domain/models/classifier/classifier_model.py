from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class PredictMoodParameters(BaseModel):
    """Parameters for mood prediction from a MIDI file"""
    midi_path: str
    model_type: str = "remi"
    device: str = "cpu"


class PredictWithCSVParameters(BaseModel):
    """Parameters for mood prediction with ground truth comparison"""
    midi_path: str
    csv_path: str
    model_type: str = "remi"
    device: str = "cpu"


class BatchPredictParameters(BaseModel):
    """Parameters for batch mood prediction from MIDI files listed in a CSV"""
    csv_path: str
    midi_dir: str
    model_type: str = "remi"
    device: str = "cpu"
    batch_size: int = 10
    output_dir: Optional[str] = None
    mood_column: Optional[str] = "mood"
    file_column: Optional[str] = "file"


class TrainingParameters(BaseModel):
    """Parameters for training a classifier model"""
    dataset: str = "ReMIDICaps"
    midi: str = "remi"
    task: str = "mood"
    labels_path: Optional[str] = "datasets/ReMIDICaps/labels"
    r: int = 8
    lstm_hidden_dim: int = 64
    d_model: int = 100  # Standardized from embedding_size
    batch_size: int = 8
    num_workers: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    T_0: int = 45
    max_epochs: int = 50
    gpus: int = 1
    distributed_backend: str = "dp"
    deterministic: bool = True
    benchmark: bool = False
    reproduce: bool = True


class MidiFeatureExtractParameters(BaseModel):
    """Parameters for extracting REMI features from MIDI files"""
    midi_path: str = "./datasets/ReMIDICaps/midi"
    remi_path: str = "./datasets/ReMIDICaps/remi_midi"
    csv_path: Optional[str] = "./datasets/ReMIDICaps/labels/ReMIDICaps_reduced.csv"
    dictionary_path: Optional[str] = "./datasets/ReMIDICaps/dictionary.pkl"


# Response models
class MoodPrediction(BaseModel):
    """Response model for mood prediction"""
    file_name: str
    dominant_mood: str
    dominant_probability: float
    mood_probabilities: Dict[str, float]


class BatchMoodPrediction(BaseModel):
    """Response model for batch mood prediction"""
    predictions: List[MoodPrediction]
    average_mood_probabilities: Dict[str, float] 