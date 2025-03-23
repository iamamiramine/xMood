from fastapi import APIRouter

from application.projection.services import projector_service

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
