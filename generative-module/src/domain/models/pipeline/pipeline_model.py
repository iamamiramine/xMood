"""
Pipeline Domain Models

This module defines the models used for pipeline orchestration and job management.
It includes models for pipeline configuration, job states, and execution tracking.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class JobStatus(str, Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ServiceType(str, Enum):
    """Available service types"""
    SYMBOLIC = "symbolic"
    LATENT = "latent"
    GENERATOR = "generator"
    MULTIMODAL = "multimodal"


class CommonParameters(BaseModel):
    """Common parameters used across multiple services"""
    
    # Highly common parameters (used by 5+ models)
    device: str = Field(default="cuda", description="Device to use for training/inference")
    batch_size: int = Field(default=32, description="Batch size for processing")
    context_size: int = Field(default=512, description="Context size for sequences")
    dataset_name: str = Field(default="EMOPIA", description="Name of the dataset")
    num_workers: int = Field(default=4, description="Number of workers for data loading")
    pin_memory: bool = Field(default=True, description="Use pinned memory for data loading")
    max_bars: int = Field(default=512, description="Maximum number of bars")
    max_positions: int = Field(default=1024, description="Maximum position embeddings")
    bar_token_idx: int = Field(default=0, description="Bar token index")
    train_val_test_split: Tuple[float, float, float] = Field(default=(0.7, 0.2, 0.1), description="Train/validation/test split")
    
    # Training parameters (used by 3-4 models)
    d_model: int = Field(default=512, description="Model dimension")
    dropout: float = Field(default=0.1, description="Dropout rate")
    lr: float = Field(default=1e-4, description="Learning rate")
    max_steps: int = Field(default=100000, description="Maximum training steps")
    warmup_steps: int = Field(default=4000, description="Warmup steps")


class ServiceSpecificParameters(BaseModel):
    """Service-specific parameters for each service"""
    
    # Dataloader specific
    dataloader: Optional[Dict[str, Any]] = Field(default=None, description="Dataloader-specific parameters")
    
    # Generator specific
    generator: Optional[Dict[str, Any]] = Field(default=None, description="Generator-specific parameters")
    
    # Latent/VAE specific
    latent: Optional[Dict[str, Any]] = Field(default=None, description="Latent/VAE-specific parameters")
    
    # Multimodal specific
    multimodal: Optional[Dict[str, Any]] = Field(default=None, description="Multimodal-specific parameters")
    
    # Symbolic specific
    symbolic: Optional[Dict[str, Any]] = Field(default=None, description="Symbolic-specific parameters")


class ParameterSetupRequest(BaseModel):
    """Request to set up parameters for pipeline configuration"""
    
    # Common parameters applied to all services
    common_parameters: CommonParameters = Field(description="Common parameters used across services")
    
    # Service-specific parameters
    service_parameters: ServiceSpecificParameters = Field(description="Service-specific parameters")
    
    # Configuration metadata
    config_name: Optional[str] = Field(default=None, description="Custom name for the configuration")
    description: Optional[str] = Field(default=None, description="Description of the configuration")
    
    # Global configuration overrides
    global_config: Optional[Dict[str, Any]] = Field(default=None, description="Global configuration overrides")
    paths_config: Optional[Dict[str, Any]] = Field(default=None, description="Path configuration overrides")
    midi_config: Optional[Dict[str, Any]] = Field(default=None, description="MIDI configuration overrides")


class ParameterSetupResponse(BaseModel):
    """Response from parameter setup"""
    
    config_id: str = Field(description="Unique configuration identifier")
    config_name: str = Field(description="Name of the configuration")
    config_file_path: str = Field(description="Path to the generated YAML configuration file")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Configuration creation timestamp")
    
    # Summary of applied parameters
    common_parameters_applied: Dict[str, Any] = Field(description="Common parameters that were applied")
    service_parameters_applied: Dict[str, Any] = Field(description="Service-specific parameters that were applied")
    
    # Validation results
    validation_passed: bool = Field(description="Whether parameter validation passed")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings")


class PipelineJobRequest(BaseModel):
    """Request to execute a pipeline job"""
    
    service_name: ServiceType = Field(
        description="Name of the service to execute"
    )
    function_name: str = Field(
        description="Name of the function to execute within the service"
    )
    job_name: Optional[str] = Field(
        default=None,
        description="Custom name for the job"
    )


class PipelineJobInfo(BaseModel):
    """Information about a pipeline job"""
    
    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique job identifier"
    )
    job_name: str = Field(
        description="Human-readable job name"
    )
    service_name: ServiceType = Field(
        description="Name of the service"
    )
    function_name: str = Field(
        description="Name of the function"
    )
    status: JobStatus = Field(
        default=JobStatus.PENDING,
        description="Current job status"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Job creation timestamp"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Job start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Job completion timestamp"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if job failed"
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Job result data"
    )
    progress: Optional[float] = Field(
        default=0.0,
        description="Job progress percentage (0-100)"
    )


class PipelineJobList(BaseModel):
    """List of pipeline jobs"""
    
    jobs: List[PipelineJobInfo] = Field(
        default_factory=list,
        description="List of pipeline jobs"
    )
    total: int = Field(
        description="Total number of jobs"
    )


class PipelineExecutionRequest(BaseModel):
    """Request to execute a complete pipeline"""
    
    pipeline_name: str = Field(
        description="Name of the pipeline to execute"
    )
    services: List[str] = Field(
        default_factory=list,
        description="List of services to execute (empty means all configured services)"
    )
    parallel_execution: bool = Field(
        default=False,
        description="Whether to execute compatible services in parallel"
    )


class PipelineExecutionResponse(BaseModel):
    """Response from pipeline execution"""
    
    pipeline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique pipeline execution identifier"
    )
    pipeline_name: str = Field(
        description="Name of the pipeline"
    )
    job_ids: List[str] = Field(
        description="List of job IDs created for this pipeline"
    )
    status: str = Field(
        default="started",
        description="Pipeline execution status"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Pipeline creation timestamp"
    ) 