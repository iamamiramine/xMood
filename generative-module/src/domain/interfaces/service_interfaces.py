"""
Service Interfaces for PA-AI-2 Generative Module

This module defines abstract base classes (interfaces) for all services in the system.
These interfaces establish contracts for service communication and enable dependency
injection without tight coupling between services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from domain.models.classifier.classifier_model import (
    PredictMoodParameters,
    TrainingParameters,
    BatchPredictParameters,
)
from domain.models.encoder.encoder_model import (
    EncodeParameters,
    EncodeDatasetParameters,
    TokenizeRemiDatasetParameters,
)
from domain.models.feature_extraction.feature_extraction_model import (
    SymbolicFeaturesParameters,
    SymbolicFeaturesDatasetParameters,
    VaeTrainingParameters,
    LatentRepresentationParameters,
)
from domain.models.generator_model import (
    GenerateFromMIDIParameters,
    GeneratorTrainingParameters,
)
from domain.models.multimodal_mapping_model import (
    MultimodalMappingParameters,
    MultimodalTrainingParameters,
)
from domain.models.music_base.music_base_model import MusicBaseParameters
from domain.models.dataloader.dataloader_model import (
    DataloaderModuleParameters,
    DatasetLoadParameters,
    DataloaderParameters,
)
from domain.models.pseudo_labeller.pseudo_labeller_model import (
    PseudoLabellerParameters,
    ImageMoodClassificationParameters,
    BatchImageMoodClassificationParameters,
)
from application.shared.services.job_management_service import JobPriority, JobStatus


class IClassifierService(ABC):
    """Interface for classifier service operations."""
    
    @abstractmethod
    def predict(self, midi_path: str, model_type: str, task: str, device: str, parameters: PredictMoodParameters) -> Dict[str, Any]:
        """Predict mood for a MIDI file."""
        pass
    
    @abstractmethod
    def predict_batch_sync(self, parameters: BatchPredictParameters) -> Dict[str, Any]:
        """Batch predict mood for multiple MIDI files."""
        pass
    
    @abstractmethod
    def train(self, parameters: TrainingParameters) -> Dict[str, Any]:
        """Train a classifier model."""
        pass


class IEncoderService(ABC):
    """Interface for encoder service operations."""
    
    @abstractmethod
    def encode_midi(self, parameters: EncodeParameters) -> Dict[str, Any]:
        """Encode a single MIDI file."""
        pass
    
    @abstractmethod
    def encode_dataset(self, parameters: EncodeDatasetParameters) -> Dict[str, Any]:
        """Encode a dataset of MIDI files."""
        pass
    
    @abstractmethod
    def tokenize_remi_dataset(self, parameters: TokenizeRemiDatasetParameters) -> Dict[str, Any]:
        """Tokenize REMI dataset."""
        pass


class IFeatureExtractionService(ABC):
    """Interface for feature extraction service operations."""
    
    @abstractmethod
    def extract_symbolic_features(self, parameters: SymbolicFeaturesParameters) -> Dict[str, Any]:
        """Extract symbolic features from a MIDI file."""
        pass
    
    @abstractmethod
    def extract_symbolic_features_dataset(self, parameters: SymbolicFeaturesDatasetParameters) -> Dict[str, Any]:
        """Extract symbolic features from a dataset."""
        pass
    
    @abstractmethod
    def train_vae(self, parameters: VaeTrainingParameters) -> Dict[str, Any]:
        """Train a VAE model."""
        pass
    
    @abstractmethod
    def generate_latent_representations_dataset(self, parameters: LatentRepresentationParameters) -> Dict[str, Any]:
        """Generate latent representations for a dataset."""
        pass


class IGeneratorService(ABC):
    """Interface for generator service operations."""
    
    @abstractmethod
    def train_generator(self, parameters: GeneratorTrainingParameters) -> Dict[str, Any]:
        """Train a generator model."""
        pass
    
    @abstractmethod
    def generate_from_midi(self, parameters: GenerateFromMIDIParameters) -> Dict[str, Any]:
        """Generate music from MIDI input."""
        pass
    
    @abstractmethod
    def batch_generate_from_dataset(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Batch generate music from dataset."""
        pass


class IMultimodalMappingService(ABC):
    """Interface for multimodal mapping service operations."""
    
    @abstractmethod
    def train_multimodal_mapping(self, parameters: MultimodalTrainingParameters) -> Dict[str, Any]:
        """Train multimodal mapping model."""
        pass
    
    @abstractmethod
    def generate_multimodal_representations(self, parameters: MultimodalMappingParameters) -> Dict[str, Any]:
        """Generate multimodal representations."""
        pass


class IMusicBaseService(ABC):
    """Interface for music base service operations."""
    
    @abstractmethod
    def extract_chords(self, parameters: MusicBaseParameters) -> Dict[str, Any]:
        """Extract chords from MIDI file."""
        pass
    
    @abstractmethod
    def synthesize_midi(self, file: str, out_dir: str) -> Dict[str, Any]:
        """Synthesize MIDI to audio."""
        pass


class IDataloaderService(ABC):
    """Interface for dataloader service operations."""
    
    @abstractmethod
    def initialize_dataloader_module(self, parameters: DataloaderModuleParameters) -> Dict[str, Any]:
        """Initialize dataloader module."""
        pass
    
    @abstractmethod
    def load_dataset(self, parameters: DatasetLoadParameters) -> Dict[str, Any]:
        """Load dataset."""
        pass
    
    @abstractmethod
    def process_dataset_files(self, parameters: DataloaderParameters) -> Dict[str, Any]:
        """Process dataset files."""
        pass


class IPseudoLabellerService(ABC):
    """Interface for pseudo labeller service operations."""
    
    @abstractmethod
    def create_mood_labels(self, parameters: PseudoLabellerParameters) -> Dict[str, Any]:
        """Create mood labels."""
        pass
    
    @abstractmethod
    def classify_image_mood(self, parameters: ImageMoodClassificationParameters) -> Dict[str, Any]:
        """Classify image mood."""
        pass
    
    @abstractmethod
    def batch_classify_images(self, parameters: BatchImageMoodClassificationParameters) -> Dict[str, Any]:
        """Batch classify image moods."""
        pass


class IConfigService(ABC):
    """Interface for configuration service operations."""
    
    @abstractmethod
    def load_config(self, config_path: str, validate: bool = True) -> Dict[str, Any]:
        """Load configuration from file."""
        pass
    
    @abstractmethod
    def get_service_config(self, service_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Get service-specific configuration."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        pass


class IJobManagementService(ABC):
    """Interface for job management service operations."""
    
    @abstractmethod
    def submit_job(self, service_name: str, function_name: str, job_function: callable, 
                   args: tuple = (), kwargs: dict = None, job_name: Optional[str] = None,
                   priority: JobPriority = JobPriority.NORMAL, pipeline_job_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit a background job."""
        pass
    
    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get job status."""
        pass
    
    @abstractmethod
    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Get job result."""
        pass
    
    @abstractmethod
    def cancel_job(self, job_id: str, reason: Optional[str] = None) -> bool:
        """Cancel a job."""
        pass


class IPipelineConfigService(ABC):
    """Interface for pipeline configuration service operations."""
    
    @abstractmethod
    def generate_pipeline_config(self, services: List[str], environment: str = "production",
                                 job_name: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate pipeline configuration."""
        pass
    
    @abstractmethod
    def get_pipeline_config(self, pipeline_job_id: str) -> Dict[str, Any]:
        """Get pipeline configuration."""
        pass
    
    @abstractmethod
    def update_pipeline_status(self, pipeline_job_id: str, status: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update pipeline status."""
        pass
    
    @abstractmethod
    def validate_pipeline_config(self, config: Dict[str, Any]) -> bool:
        """Validate pipeline configuration."""
        pass 