from fastapi import APIRouter

from domain.models.captioning.captioning_model import CaptionDatasetParameters
from application.projection.services import projector_service
from application.projection.services import captioning_service

router = APIRouter()


@router.post("/train_projector")
def train_projector(config_path: str) -> dict:
    """
    Description:
    ------------
        Train Projector

    Parameters:
    -----------
        config_path: str

    Returns:
    --------
    dict
        A dictionary

    """
    return projector_service.train_projector(config_path)


@router.post("/generate_features_from_prompt")
def generate_features_from_prompt(config_path: str) -> dict:
    """
    Description:
    ------------
        Generate Features From Prompt

    Parameters:
    -----------
        config_path: str

    Returns:
    --------
    dict
        A dictionary

    """
    return projector_service.generate_features_from_prompt(config_path)


@router.post("/caption_dataset")
def caption_dataset(parameters: CaptionDatasetParameters) -> dict:
    """
    Description:
    ------------
        Preprocess Dataset Emotions

    Parameters:
    -----------
        parameters: CaptionDatasetParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return captioning_service.caption_dataset(parameters)
