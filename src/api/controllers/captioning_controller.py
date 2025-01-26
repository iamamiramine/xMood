from fastapi import APIRouter

from domain.models.captioning.captioning_model import CaptionDatasetParameters
from application.captioning.services import captioning_service

router = APIRouter()


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
