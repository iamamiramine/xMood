"""Placeholder adapter for music base service."""
from typing import Dict, Any
from domain.interfaces.service_interfaces import IMusicBaseService
from domain.models.music_base.music_base_model import *
from application.music_base.services import music_base_service

class MusicBaseServiceAdapter(IMusicBaseService):
    def extract_chords(self, parameters) -> Dict[str, Any]:
        return music_base_service.extract_chords(parameters)
    
    def synthesize_midi(self, file: str, out_dir: str) -> Dict[str, Any]:
        return music_base_service.synthesize_midi(file, out_dir) 