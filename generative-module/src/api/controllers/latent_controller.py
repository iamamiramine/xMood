import asyncio

from fastapi import APIRouter

from application.latent.services.latent_service import (
    train_vae as _train_vae,
    generate_latent_representations_dataset as _generate_latent_representations_dataset,
)
from domain.models.latent.latent_model import (
    VaeTrainingParameters,
    LatentRepresentationParameters,
)

router = APIRouter()


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
