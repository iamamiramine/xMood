import asyncio

from fastapi import APIRouter

from domain.models.feature_extraction.feature_extraction_model import SymbolicFeaturesParameters

from application.feature_extraction.services import feature_extraction_service

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
def extract_symbolic_features_dataset(dataset_name: str, level: str = "bar", add_position_tokens: bool = False) -> dict:
    """
    Description:
    ------------
        Extract Symbolic Features for Dataset

    Parameters:
    -----------
        dataset_name: str

    Returns:
    --------
    dict
        A dictionary

    """
    return asyncio.run(feature_extraction_service.extract_symbolic_features_dataset(dataset_name, level, add_position_tokens))


@router.post("/train_vae")
def train_vae(config_path: str) -> dict:
    """
    Description:
    ------------
        Train VAE

    Parameters:
    -----------
        config_path: str

    Returns:
    --------
    dict
        A dictionary

    """
    return feature_extraction_service.train_vae(config_path)


@router.post("/generate_latent_representations_dataset")
def generate_latent_representations_dataset(config_path: str) -> dict:
    """
    Description:
    ------------
        Generate Latent Representations of a Dataset

    Parameters:
    -----------
        config_path: str

    Returns:
    --------
    dict
        A dictionary

    """
    return feature_extraction_service.generate_latent_representations_dataset(config_path)
