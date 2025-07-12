"""Placeholder adapter for generator service."""
from typing import Dict, Any
from domain.interfaces.service_interfaces import IGeneratorService
from domain.models.generator_model import *
from application.generator.services import generator_service

class GeneratorServiceAdapter(IGeneratorService):
    def train_generator(self, parameters) -> Dict[str, Any]:
        return generator_service.train_generator(parameters)
    
    def generate_from_midi(self, parameters) -> Dict[str, Any]:
        return generator_service.generate_from_midi(parameters)
    
    def batch_generate_from_dataset(self, parameters) -> Dict[str, Any]:
        return generator_service.batch_generate_from_dataset(parameters) 