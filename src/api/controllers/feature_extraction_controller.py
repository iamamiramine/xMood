import asyncio

from fastapi import APIRouter

from src.domain.models.feature_extraction.feature_extraction_model import (
    VQVAEParameters,
    LatentRepresentationParameters,
    TSNEParameters,
    CorrelationParameters,
    LatentRepresentationDatasetParameters,
    DescriptionParameters,
    DescriptionDatasetParameters,
)

from src.application.feature_extraction.services import feature_extraction_service

router = APIRouter()


@router.post("/extract_description_sample")
def extract_description_sample(parameters: DescriptionParameters) -> dict:
    """
    Description:
    ------------
        Extract Description for 1 File

    Parameters:
    -----------
        parameters: DescriptionParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return feature_extraction_service.extract_description_sample(parameters)


@router.post("/extract_description_dataset")
def extract_description_dataset(parameters: DescriptionDatasetParameters) -> dict:
    """
    Description:
    ------------
        Extract Description for Dataset

    Parameters:
    -----------
        parameters: DescriptionDatasetParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return asyncio.run(feature_extraction_service.extract_description_dataset(parameters))


@router.post("/train_vae")
def train_vae(parameters: VQVAEParameters) -> dict:
    """
    Description:
    ------------
        Train VAE

    Parameters:
    -----------
        parameters: VQVAEParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return feature_extraction_service.train_vae(parameters)


@router.post("/generate_latent_representations_dataset")
def generate_latent_representations_dataset(parameters: LatentRepresentationDatasetParameters) -> dict:
    """
    Description:
    ------------
        Generate Latent Representations of a Dataset

    Parameters:
    -----------
        parameters: VQVAEParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return feature_extraction_service.generate_latent_representations_dataset(parameters)


@router.post("/generate_tsne")
def generate_tsne(parameters: TSNEParameters) -> dict:
    """
    Description:
    ------------
        Generate Latent Representations of a File

    Parameters:
    -----------
        parameters: LatentRepresentationParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return feature_extraction_service.generate_tsne(parameters)


@router.post("/get_correlation")
def get_correlation(parameters: CorrelationParameters) -> dict:
    """
    Description:
    ------------
        Generate Latent Representations of a File

    Parameters:
    -----------
        parameters: LatentRepresentationParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return feature_extraction_service.get_correlation(parameters)
