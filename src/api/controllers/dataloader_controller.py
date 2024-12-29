from fastapi import APIRouter

from src.domain.models.dataloader.dataloader_model import DataloaderParameters
from src.application.dataloader.services import dataloader_service

router = APIRouter()


@router.post("/prepare_dataloader")
def prepare_dataloader(parameters: DataloaderParameters) -> dict:
    """
    Description:
    ------------
        Prepare dataloader for training

    Parameters:
    -----------
        parameters: DataloaderParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return dataloader_service.prepare_dataloader(parameters)


@router.post("/run_dataloader")
def run_dataloader(dataset_name: str, load_latent: bool = False, load_desc: bool = False) -> dict:
    """
    Description:
    ------------
        Run dataloader

    Parameters:
    -----------
        parameters: dataset_name

    Returns:
    --------
    dict
        A dictionary

    """
    return dataloader_service.run_dataloader(dataset_name, load_latent, load_desc)
