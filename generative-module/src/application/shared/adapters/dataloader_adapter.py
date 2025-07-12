"""Placeholder adapter for dataloader service."""
from typing import Dict, Any
from domain.interfaces.service_interfaces import IDataloaderService
from domain.models.dataloader.dataloader_model import *
from application.dataloader.services import dataloader_service

class DataloaderServiceAdapter(IDataloaderService):
    def initialize_dataloader_module(self, parameters) -> Dict[str, Any]:
        return dataloader_service.initialize_dataloader_module(parameters)
    
    def load_dataset(self, parameters) -> Dict[str, Any]:
        return dataloader_service.load_dataset(parameters)
    
    def process_dataset_files(self, parameters) -> Dict[str, Any]:
        return dataloader_service.process_dataset_files(parameters) 