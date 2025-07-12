"""Placeholder adapter for feature extraction service."""
from typing import Dict, Any
from domain.interfaces.service_interfaces import IFeatureExtractionService
from domain.models.feature_extraction.feature_extraction_model import *
from application.feature_extraction.services import feature_extraction_service

class FeatureExtractionServiceAdapter(IFeatureExtractionService):
    def extract_symbolic_features(self, parameters) -> Dict[str, Any]:
        return feature_extraction_service.extract_symbolic_features(parameters)
    
    def extract_symbolic_features_dataset(self, parameters) -> Dict[str, Any]:
        return feature_extraction_service.extract_symbolic_features_dataset(parameters)
    
    def train_vae(self, parameters) -> Dict[str, Any]:
        return feature_extraction_service.train_vae(parameters)
    
    def generate_latent_representations_dataset(self, parameters) -> Dict[str, Any]:
        return feature_extraction_service.generate_latent_representations_dataset(parameters) 