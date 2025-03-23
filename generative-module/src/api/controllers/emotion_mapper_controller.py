from fastapi import APIRouter

from domain.models.emotion_mapper.emotion_mapper_model import (
    EmotionMapperTrainingParameters,
    EmotionMapperGenerateParameters,
)
from application.emotion_mapper.services import emotion_mapper_service

router = APIRouter()


@router.post("/extract_emotion_vectors")
def extract_emotion_vectors(dataset_name: str) -> dict:
    """
    Description:
    ------------
        Extract emotion vectors for all MIDI files in a dataset

    Parameters:
    -----------
        dataset_name: str
            Name of the dataset to process

    Returns:
    --------
    dict
        A dictionary containing the extraction status message
    """
    return emotion_mapper_service.extract_emotion_vectors(dataset_name)


@router.post("/train_emotion_mapper")
def train_emotion_mapper(parameters: EmotionMapperTrainingParameters) -> dict:
    """
    Description:
    ------------
        Train Emotion Mapper

    Parameters:
    -----------
        parameters: EmotionMapperTrainingParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return emotion_mapper_service.train_emotion_mapper(parameters)


@router.post("/generate_bar_emotions")
def generate_bar_emotions(parameters: EmotionMapperGenerateParameters) -> dict:
    """
    Description:
    ------------
        Generate Bar Emotions

    Parameters:
    -----------
        parameters: EmotionMapperGenerateParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return emotion_mapper_service.generate_bar_emotions(parameters)
