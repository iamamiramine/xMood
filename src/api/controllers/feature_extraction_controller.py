import asyncio

from fastapi import APIRouter

from src.domain.models.feature_extraction.feature_extraction_model import (
    VQVAEParameters,
    LatentRepresentationParameters,
    TSNEParameters,
    CorrelationParameters,
    LatentRepresentationDatasetParameters,
    SymbolicFeaturesParameters,
    SymbolicFeaturesDatasetParameters,
)

from src.application.feature_extraction.services import feature_extraction_service

router = APIRouter()


@router.post("/extract_symbolic_features")
def extract_symbolic_features(parameters: SymbolicFeaturesParameters) -> dict:
    """
    Description:
    ------------
        Extract Symbolic Features for 1 File

    Parameters:
    -----------
        parameters: DescriptionParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return feature_extraction_service.extract_symbolic_features(parameters)


@router.post("/extract_symbolic_features_dataset")
def extract_symbolic_features_dataset(parameters: SymbolicFeaturesDatasetParameters) -> dict:
    """
    Description:
    ------------
        Extract Symbolic Features for Dataset

    Parameters:
    -----------
        parameters: DescriptionDatasetParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return asyncio.run(feature_extraction_service.extract_symbolic_features_dataset(parameters))


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
