from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from typing import Dict, Any
import os
import tempfile
import shutil

from application.classifier.services import classifier_service
from application.shared.helpers import enum_helpers
from domain.models.classifier.classifier_model import (
    PredictMoodParameters,
    BatchPredictParameters,
    TrainingParameters,
)

# Create router
router = APIRouter()


@router.post("/predict", response_model=Dict[str, Any])
def predict(parameters: PredictMoodParameters) -> dict:
    """
    Description:
    ------------
        Predict mood for a MIDI file

    Parameters:
    -----------
        parameters: PredictMoodParameters
            Parameters for prediction including:
            - midi_path: Path to the MIDI file
            - model_type: Model type (remi, midi_like, wav)
            - device: Device to run inference on (cpu, cuda:0)

    Returns:
    --------
    dict
        A dictionary containing prediction results
    """
    return classifier_service.predict(parameters.midi_path, parameters.model_type, "mood", parameters.device, parameters)


@router.post("/predict_batch", response_model=Dict[str, Any])
def predict_batch(parameters: BatchPredictParameters) -> dict:
    """
    Description:
    ------------
        Batch predict mood for MIDI files listed in a CSV and compare with ground truth

    Parameters:
    -----------
        parameters: BatchPredictParameters
            Parameters for batch prediction including:
            - csv_path: Path to CSV containing MIDI filenames and ground truth
            - midi_dir: Directory containing MIDI files
            - model_type: Model type (remi, midi_like, wav)
            - batch_size: Number of files to process in parallel
            - device: Device to run inference on
            - output_dir: Directory to save prediction results (optional)
            - mood_column: Name of the column containing ground truth moods (optional, default: "mood_tokens")
            - file_column: Name of the column containing filenames (optional, default: "file")

    Returns:
    --------
    dict
        A dictionary containing prediction results and statistics
    """
    # Apply default values for parameters
    enum_helpers.validate_parameters_defaults(parameters, {
        'mood_column': 'mood_tokens'
    })
    
    return classifier_service.predict_batch_sync(parameters)


@router.post("/train_model", response_model=Dict[str, Any])
def train_model(parameters: TrainingParameters) -> dict:
    """
    Description:
    ------------
        Train a classifier model with the specified parameters

    Parameters:
    -----------
        parameters: TrainingParameters
            Training configuration parameters

    Returns:
    --------
    dict
        A dictionary with training status
    """
    return classifier_service.train(parameters)
