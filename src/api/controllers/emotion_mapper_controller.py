from fastapi import APIRouter

from domain.models.emotion_mapper.emotion_mapper_model import (
    EmotionMapperTrainingParameters,
    EmotionMapperGenerateParameters,
)
from application.emotion_mapper.services import emotion_mapper_service

router = APIRouter()


@router.post("/preprocess_dataset_emotions")
def preprocess_dataset_emotions(dataset_name: str) -> dict:
    """
    Description:
    ------------
        Preprocess Dataset Emotions

    Parameters:
    -----------
        dataset_name: str

    Returns:
    --------
    dict
        A dictionary

    """
    return emotion_mapper_service.preprocess_dataset_emotions(dataset_name)


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
