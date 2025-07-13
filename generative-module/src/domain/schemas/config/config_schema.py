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
from domain.models.symbolic.symbolic_model import (
    MusicBaseParameters,
    EncodeParameters,
    EncodeDatasetParameters, 
    TokenizeRemiDatasetParameters
)
from domain.models.latent.latent_model import (
    VaeTrainingParameters,
    LatentRepresentationParameters
)
from domain.models.generator.generator_model import (
    GenerateFromMIDIParameters,
    GeneratorTrainingParameters
)
from domain.models.multimodal.multimodal_model import (
    MultimodalMappingParameters,
    MultimodalTrainingParameters
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


class SymbolicServiceConfig(BaseConfig):
    """Configuration for symbolic service operations"""

    # Configuration for synthesize_midi function
    synthesize_midi: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for synthesize_midi function"
    )
    
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


class LatentServiceConfig(BaseConfig):
    """Configuration for feature extraction service operations"""
    
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


class MultimodalServiceConfig(BaseConfig):
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
    symbolic_service: SymbolicServiceConfig = Field(default_factory=SymbolicServiceConfig)
    latent_service: LatentServiceConfig = Field(default_factory=LatentServiceConfig)
    generator_service: GeneratorServiceConfig = Field(default_factory=GeneratorServiceConfig)
    multimodal_service: MultimodalServiceConfig = Field(default_factory=MultimodalServiceConfig)
    
    def get_config_for_service(self, service_name: str) -> Optional[BaseConfig]:
        """Get configuration for a specific service"""
        service_configs = {
            "symbolic": self.symbolic_service,
            "latent": self.latent_service,
            "generator": self.generator_service,
            "multimodal": self.multimodal_service,
            "paths": self.paths,
            "midi": self.midi,
            "global": self.global_config,
        }
        return service_configs.get(service_name)


# Parameter model registry that maps service functions to their parameter models
# This is used by the pipeline_job_service to parse configuration to parameters
PARAMETER_MODEL_REGISTRY = {
    "symbolic": {
        "synthesize_midi": MusicBaseParameters,
        "encode_midi": EncodeParameters,
        "encode_dataset": EncodeDatasetParameters,
        "tokenize_remi_dataset": TokenizeRemiDatasetParameters,
    },
    "latent": {
        "train_vae": VaeTrainingParameters,
        "generate_latent_representations": LatentRepresentationParameters,
    },
    "generator": {
        "train": GeneratorTrainingParameters,
        "generate_from_midi": GenerateFromMIDIParameters,
    },
    "multimodal": {
        "train": MultimodalTrainingParameters,
        "generate_representations": MultimodalMappingParameters,
    },
} 