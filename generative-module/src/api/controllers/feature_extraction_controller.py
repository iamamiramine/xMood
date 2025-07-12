import asyncio

from fastapi import APIRouter

from application.feature_extraction.services.feature_extraction_service import (
    extract_symbolic_features as _extract_symbolic_features,
    extract_symbolic_features_dataset as _extract_symbolic_features_dataset,
    train_vae as _train_vae,
    generate_latent_representations_dataset as _generate_latent_representations_dataset,
)
from domain.models.feature_extraction.feature_extraction_model import (
    SymbolicFeaturesParameters,
    SymbolicFeaturesDatasetParameters,
    VaeTrainingParameters,
    LatentRepresentationParameters,
)

router = APIRouter()


@router.post("/extract_symbolic_features")
def extract_symbolic_features(parameters: SymbolicFeaturesParameters) -> dict:
    """
    Extract symbolic features from a single MIDI file.
    
    Args:
        parameters: SymbolicFeaturesParameters containing file path and extraction options
        
    Returns:
        dict: Extraction result message
    """
    try:
        result = _extract_symbolic_features(parameters)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/extract_symbolic_features_dataset")
async def extract_symbolic_features_dataset(parameters: SymbolicFeaturesDatasetParameters) -> dict:
    """
    Extract symbolic features from an entire dataset of MIDI files.
    
    Args:
        parameters: SymbolicFeaturesDatasetParameters containing dataset configuration
        
    Returns:
        dict: Extraction result summary
    """
    try:
        result = await _extract_symbolic_features_dataset(parameters)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/train_vae")
def train_vae(parameters: VaeTrainingParameters) -> dict:
    """
    Train a VAE model for latent representation learning.
    
    Args:
        parameters: VaeTrainingParameters containing training configuration
        
    Returns:
        dict: Training result message
    """
    try:
        result = _train_vae(parameters)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/generate_latent_representations_dataset")
async def generate_latent_representations_dataset(parameters: LatentRepresentationParameters) -> dict:
    """
    Generate latent representations for an entire dataset using a trained VAE.
    
    Args:
        parameters: LatentRepresentationParameters containing generation configuration
        
    Returns:
        dict: Generation result summary
    """
    try:
        result = await _generate_latent_representations_dataset(parameters)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
