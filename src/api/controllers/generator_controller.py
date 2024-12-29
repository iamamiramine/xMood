from fastapi import APIRouter

from src.domain.models.generator.generator_model import GeneratorTrainingParameters, GeneratorGenerateParameters, GeneratorGeneratePromptParameters
from src.application.generator.services import generator_service

router = APIRouter()


@router.post("/train_generator")
def train_generator(parameters: GeneratorTrainingParameters) -> dict:
    """
    Description:
    ------------
        Train Generator

    Parameters:
    -----------
        parameters: GeneratorTrainingParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return generator_service.train_generator(parameters)


@router.post("/generate_sample_from_prompt")
def generate_sample_from_prompt(parameters: GeneratorGeneratePromptParameters) -> dict:
    """
    Description:
    ------------
        Generate Sample from Prompt

    Parameters:
    -----------
        parameters: GeneratorGeneratePromptParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return generator_service.generate_sample_from_prompt(parameters)
