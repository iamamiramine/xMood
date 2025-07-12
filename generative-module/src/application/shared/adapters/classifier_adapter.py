"""
Classifier Service Adapter

This adapter implements the IClassifierService interface for the existing
classifier service, enabling dependency injection and reducing cross-service
dependencies.
"""

from typing import Dict, Any
import logging

from domain.interfaces.service_interfaces import IClassifierService
from domain.models.classifier.classifier_model import (
    PredictMoodParameters,
    TrainingParameters,
    BatchPredictParameters,
)

# Import the actual classifier service functions
from application.classifier.services import classifier_service

logger = logging.getLogger(__name__)


class ClassifierServiceAdapter(IClassifierService):
    """
    Adapter for the classifier service that implements the IClassifierService interface.
    
    This adapter wraps the existing classifier service functions to provide
    a clean interface for dependency injection.
    """
    
    def __init__(self):
        logger.info("ClassifierServiceAdapter initialized")
    
    def predict(self, midi_path: str, model_type: str, task: str, device: str, 
                parameters: PredictMoodParameters) -> Dict[str, Any]:
        """
        Predict mood for a MIDI file.
        
        Args:
            midi_path: Path to the MIDI file
            model_type: Type of model to use
            task: Task type
            device: Device to run inference on
            parameters: Prediction parameters
            
        Returns:
            Prediction results
        """
        try:
            return classifier_service.predict(midi_path, model_type, task, device, parameters)
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            return {"status": "error", "message": str(e)}
    
    def predict_batch_sync(self, parameters: BatchPredictParameters) -> Dict[str, Any]:
        """
        Batch predict mood for multiple MIDI files.
        
        Args:
            parameters: Batch prediction parameters
            
        Returns:
            Batch prediction results
        """
        try:
            return classifier_service.predict_batch_sync(parameters)
        except Exception as e:
            logger.error(f"Error in predict_batch_sync: {e}")
            return {"status": "error", "message": str(e)}
    
    def train(self, parameters: TrainingParameters) -> Dict[str, Any]:
        """
        Train a classifier model.
        
        Args:
            parameters: Training parameters
            
        Returns:
            Training results
        """
        try:
            return classifier_service.train(parameters)
        except Exception as e:
            logger.error(f"Error in train: {e}")
            return {"status": "error", "message": str(e)} 