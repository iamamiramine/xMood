from fastapi import APIRouter

from application.generator.services.generator_service import (
    train_generator as _train_generator,
    generate_from_midi as _generate_from_midi,
    batch_generate_from_dataset as _batch_generate_from_dataset,
    save_checkpoint_separate as _save_checkpoint_separate,
)
from domain.models.generator_model import GenerateFromMIDIParameters, GeneratorTrainingParameters

router = APIRouter()


@router.post("/train_generator")
def train_generator(parameters: GeneratorTrainingParameters) -> dict:
    """
    Train a generator model for MIDI generation.
    
    Args:
        parameters: GeneratorTrainingParameters containing training configuration
        
    Returns:
        dict: Training result message
    """
    try:
        result = _train_generator(parameters)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/generate_from_midi")
def generate_from_midi(parameters: GenerateFromMIDIParameters) -> dict:
    """
    Generate new MIDI sequences from existing MIDI files.
    
    Args:
        parameters: GenerateFromMIDIParameters containing generation configuration
        
    Returns:
        dict: Generation result message
    """
    try:
        result = _generate_from_midi(parameters)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/batch_generate_from_dataset")
def batch_generate_from_dataset(dataset_csv_path: str, parameters: GenerateFromMIDIParameters) -> dict:
    """
    Generate MIDI sequences for an entire dataset.
    
    Args:
        dataset_csv_path: Path to the dataset CSV file
        parameters: GenerateFromMIDIParameters containing generation configuration
        
    Returns:
        dict: Batch generation result summary
    """
    try:
        result = _batch_generate_from_dataset(dataset_csv_path, parameters)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/save_checkpoint_separate")
def save_checkpoint_separate(checkpoint_path: str, output_dir: str) -> dict:
    """
    Save checkpoint components separately for easier deployment.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        output_dir: Directory to save separated components
        
    Returns:
        dict: Save operation result message
    """
    try:
        result = _save_checkpoint_separate(checkpoint_path, output_dir)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
