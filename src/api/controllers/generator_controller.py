from fastapi import APIRouter

from application.generator.services import generator_service

router = APIRouter()


@router.post("/train_generator")
def train_generator(config_path: str) -> dict:
    """
    Description:
    ------------
        Train Generator

    Parameters:
    -----------
        config_path: str

    Returns:
    --------
    dict
        A dictionary

    """
    return generator_service.train_generator(config_path)


@router.post("/generate_sample_from_prompt")
def generate_sample_from_prompt(config_path: str) -> dict:
    """
    Description:
    ------------
        Generate Sample from Prompt

    Parameters:
    -----------
        config_path: str

    Returns:
    --------
    dict
        A dictionary

    """
    return generator_service.generate_sample_from_prompt(config_path)
