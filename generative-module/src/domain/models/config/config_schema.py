"""
Configuration Schema for PA-AI Generative Module

This module defines the unified configuration structure that maps to the 
BaseModel parameters used by the pipeline services. It imports the parameter
models from the domain.models package and creates a FullConfig that can be
parsed from YAML configuration files.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# Import parameter models from domain.models
from domain.models.encoder.encoder_model import (
    EncodeParameters,
    EncodeDatasetParameters, 
    TokenizeRemiDatasetParameters
)
from domain.models.feature_extraction.feature_extraction_model import (
    SymbolicFeaturesParameters,
    SymbolicFeaturesDatasetParameters,
    VaeTrainingParameters,
    LatentRepresentationParameters
)
from domain.models.generator_model import (
    GenerateFromMIDIParameters,
    GeneratorTrainingParameters
)
from domain.models.multimodal_mapping_model import (
    MultimodalMappingParameters,
    MultimodalTrainingParameters
)
from domain.models.music_base.music_base_model import MusicBaseParameters
from domain.models.dataloader.dataloader_model import (
    DataloaderModuleParameters,
    DatasetLoadParameters,
    DataloaderParameters
)


class BaseConfig(BaseModel):
    """Base configuration with common parameters"""
    
    class Config:
        # Allow arbitrary types for enum compatibility
        arbitrary_types_allowed = True
        # Use enum values in JSON serialization
        use_enum_values = True


class PathsConfig(BaseConfig):
    """Configuration for paths and directories"""
    
    ROOT_OUTPUT: str = Field(
        default="output",
        description="Root output directory for all generated files"
    )
    DATASETS_PATH: str = Field(
        default="datasets",
        description="Path to datasets directory"
    )
    DATASET_NAME: str = Field(
        default="EMOPIA",
        description="Name of the dataset to use"
    )


class MidiConfig(BaseConfig):
    """Configuration for MIDI processing parameters"""
    
    pos_per_quarter: int = Field(
        default=12,
        description="Number of positions per quarter note",
        ge=1, le=64
    )
    resolution: int = Field(
        default=480,
        description="MIDI resolution in ticks per quarter note",
        ge=120, le=960
    )
    max_bar_length: int = Field(
        default=3,
        description="Maximum bar length multiplier",
        ge=1, le=8
    )
    max_bars: int = Field(
        default=512,
        description="Maximum number of bars in a piece",
        ge=1, le=2048
    )
    pad_idx: int = Field(
        default=338,
        description="Padding token index",
        ge=0
    )
    feature_path: str = Field(
        default="output/demos/demo_2/processed/ReMIDICaps_tokenized/",
        description="Path to MIDI feature files"
    )


class GlobalConfig(BaseConfig):
    """Global configuration settings"""
    
    device: str = Field(
        default="cuda",
        description="Device to use for training/inference"
    )
    seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    output_root: str = Field(
        default="output/",
        description="Root output directory"
    )
    checkpoints_root: str = Field(
        default="checkpoints/",
        description="Root checkpoints directory"
    )
    logs_root: str = Field(
        default="logs/",
        description="Root logs directory"
    )


class PipelineMetadata(BaseConfig):
    """Pipeline metadata configuration"""
    
    version: str = Field(
        default="1.0",
        description="Configuration version"
    )
    schema: str = Field(
        default="pa-ai-unified-config-v1",
        description="Configuration schema"
    )
    description: str = Field(
        default="Unified configuration for PA-AI-2 generative module",
        description="Configuration description"
    )
    generated_by: str = Field(
        default="PipelineConfigService",
        description="Generator of the configuration"
    )


class EncoderServiceConfig(BaseConfig):
    """Configuration for encoder service operations"""
    
    # Configuration for encode_midi function
    encode_midi: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for encode_midi function"
    )
    
    # Configuration for encode_dataset function
    encode_dataset: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for encode_dataset function"
    )
    
    # Configuration for tokenize_remi_dataset function
    tokenize_remi_dataset: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for tokenize_remi_dataset function"
    )


class FeatureExtractionServiceConfig(BaseConfig):
    """Configuration for feature extraction service operations"""
    
    # Configuration for extract_symbolic_features function
    extract_symbolic_features: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for extract_symbolic_features function"
    )
    
    # Configuration for extract_symbolic_features_dataset function
    extract_symbolic_features_dataset: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for extract_symbolic_features_dataset function"
    )
    
    # Configuration for train_vae function
    train_vae: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for train_vae function"
    )
    
    # Configuration for generate_latent_representations function
    generate_latent_representations: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for generate_latent_representations function"
    )


class GeneratorServiceConfig(BaseConfig):
    """Configuration for generator service operations"""
    
    # Configuration for train function
    train: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for train function"
    )
    
    # Configuration for generate_from_midi function
    generate_from_midi: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for generate_from_midi function"
    )
    
    # Configuration for batch_generate function
    batch_generate: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for batch_generate function"
    )


class MultimodalMappingServiceConfig(BaseConfig):
    """Configuration for multimodal mapping service operations"""
    
    # Configuration for train function
    train: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for train function"
    )
    
    # Configuration for generate_representations function
    generate_representations: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for generate_representations function"
    )


class MusicBaseServiceConfig(BaseConfig):
    """Configuration for music base service operations"""
    
    # Configuration for extract_chords function
    extract_chords: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for extract_chords function"
    )
    
    # Configuration for synthesize_midi function
    synthesize_midi: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for synthesize_midi function"
    )


class DataloaderServiceConfig(BaseConfig):
    """Configuration for dataloader service operations"""
    
    # Configuration for initialize_module function
    initialize_module: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for initialize_module function"
    )
    
    # Configuration for load_dataset function
    load_dataset: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for load_dataset function"
    )
    
    # Configuration for process_files function
    process_files: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for process_files function"
    )


class FullConfig(BaseConfig):
    """Complete configuration combining all components"""
    
    # Metadata
    metadata: PipelineMetadata = Field(default_factory=PipelineMetadata)
    
    # Global settings
    global_config: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")
    
    # Path configurations
    paths: PathsConfig = Field(default_factory=PathsConfig)
    
    # MIDI configurations
    midi: MidiConfig = Field(default_factory=MidiConfig)
    
    # Service configurations - these map to the parameter models used by pipeline_job_service
    encoder: EncoderServiceConfig = Field(default_factory=EncoderServiceConfig)
    feature_extraction: FeatureExtractionServiceConfig = Field(default_factory=FeatureExtractionServiceConfig)
    generator: GeneratorServiceConfig = Field(default_factory=GeneratorServiceConfig)
    multimodal_mapping: MultimodalMappingServiceConfig = Field(default_factory=MultimodalMappingServiceConfig)
    music_base: MusicBaseServiceConfig = Field(default_factory=MusicBaseServiceConfig)
    dataloader: DataloaderServiceConfig = Field(default_factory=DataloaderServiceConfig)
    
    def get_config_for_service(self, service_name: str) -> Optional[BaseConfig]:
        """Get configuration for a specific service"""
        service_configs = {
            "encoder": self.encoder,
            "feature_extraction": self.feature_extraction,
            "generator": self.generator,
            "multimodal_mapping": self.multimodal_mapping,
            "music_base": self.music_base,
            "dataloader": self.dataloader,
            "paths": self.paths,
            "midi": self.midi,
            "global": self.global_config,
        }
        return service_configs.get(service_name)


# Parameter model registry that maps service functions to their parameter models
# This is used by the pipeline_job_service to parse configuration to parameters
PARAMETER_MODEL_REGISTRY = {
    "encoder": {
        "encode_midi": EncodeParameters,
        "encode_dataset": EncodeDatasetParameters,
        "tokenize_remi_dataset": TokenizeRemiDatasetParameters,
    },
    "feature_extraction": {
        "extract_symbolic_features": SymbolicFeaturesParameters,
        "extract_symbolic_features_dataset": SymbolicFeaturesDatasetParameters,
        "train_vae": VaeTrainingParameters,
        "generate_latent_representations": LatentRepresentationParameters,
    },
    "generator": {
        "train": GeneratorTrainingParameters,
        "generate_from_midi": GenerateFromMIDIParameters,
    },
    "multimodal_mapping": {
        "train": MultimodalTrainingParameters,
        "generate_representations": MultimodalMappingParameters,
    },
    "music_base": {
        "extract_chords": MusicBaseParameters,
        "synthesize_midi": MusicBaseParameters,
    },
    "dataloader": {
        "initialize_module": DataloaderModuleParameters,
        "load_dataset": DatasetLoadParameters,
        "process_files": DataloaderParameters,
    },
} 