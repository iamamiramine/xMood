"""
Encoder Service Adapter

This adapter implements the IEncoderService interface for the existing
encoder service, enabling dependency injection and reducing cross-service
dependencies.
"""

from typing import Dict, Any
import logging

from domain.interfaces.service_interfaces import IEncoderService
from domain.models.encoder.encoder_model import (
    EncodeParameters,
    EncodeDatasetParameters,
    TokenizeRemiDatasetParameters,
)
from application.encoder.services import encoder_service

logger = logging.getLogger(__name__)


class EncoderServiceAdapter(IEncoderService):
    """
    Adapter for the encoder service that implements the IEncoderService interface.
    """
    
    def __init__(self):
        logger.info("EncoderServiceAdapter initialized")
    
    def encode_midi(self, parameters: EncodeParameters) -> Dict[str, Any]:
        """Encode a single MIDI file."""
        try:
            return encoder_service.encode_midi(parameters)
        except Exception as e:
            logger.error(f"Error in encode_midi: {e}")
            return {"status": "error", "message": str(e)}
    
    def encode_dataset(self, parameters: EncodeDatasetParameters) -> Dict[str, Any]:
        """Encode a dataset of MIDI files."""
        try:
            return encoder_service.encode_dataset(parameters)
        except Exception as e:
            logger.error(f"Error in encode_dataset: {e}")
            return {"status": "error", "message": str(e)}
    
    def tokenize_remi_dataset(self, parameters: TokenizeRemiDatasetParameters) -> Dict[str, Any]:
        """Tokenize REMI dataset."""
        try:
            return encoder_service.tokenize_remi_dataset(parameters)
        except Exception as e:
            logger.error(f"Error in tokenize_remi_dataset: {e}")
            return {"status": "error", "message": str(e)} 