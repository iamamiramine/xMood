"""Placeholder adapter for pseudo labeller service."""
from typing import Dict, Any
from domain.interfaces.service_interfaces import IPseudoLabellerService
from domain.models.pseudo_labeller.pseudo_labeller_model import *
from application.psuedo_labeller.services import pseudo_labeller_service

class PseudoLabellerServiceAdapter(IPseudoLabellerService):
    def create_mood_labels(self, parameters) -> Dict[str, Any]:
        return pseudo_labeller_service.create_mood_labels(parameters)
    
    def classify_image_mood(self, parameters) -> Dict[str, Any]:
        return pseudo_labeller_service.classify_image_mood(parameters)
    
    def batch_classify_images(self, parameters) -> Dict[str, Any]:
        return pseudo_labeller_service.batch_classify_images(parameters) 