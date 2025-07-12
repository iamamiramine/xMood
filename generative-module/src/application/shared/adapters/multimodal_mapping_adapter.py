"""Placeholder adapter for multimodal mapping service."""
from typing import Dict, Any
from domain.interfaces.service_interfaces import IMultimodalMappingService
from domain.models.multimodal_mapping_model import *
from application.multimodal_mapping.services import multimodal_mapping_service

class MultimodalMappingServiceAdapter(IMultimodalMappingService):
    def train_multimodal_mapping(self, parameters) -> Dict[str, Any]:
        return multimodal_mapping_service.train_multimodal_mapping(parameters)
    
    def generate_multimodal_representations(self, parameters) -> Dict[str, Any]:
        return multimodal_mapping_service.generate_multimodal_representations(parameters) 