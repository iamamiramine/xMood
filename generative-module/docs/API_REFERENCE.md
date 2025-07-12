# API Reference - Service Interfaces

## Overview

This document provides detailed API specifications for all service interfaces in the PA-AI-2 generative module.

## Core Services

### IConfigService

```python
from typing import Dict, Any, Optional

class IConfigService:
    """Configuration management service interface."""
    
    def load_config(self, config_path: str, validate: bool = True) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            validate: Whether to validate configuration
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationException: If configuration is invalid
            FileNotFoundError: If configuration file not found
        """
        pass
    
    def get_service_config(self, service_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get service-specific configuration.
        
        Args:
            service_name: Name of the service
            config_path: Optional path to configuration file
            
        Returns:
            Service configuration dictionary
            
        Raises:
            ConfigurationException: If service not found in configuration
        """
        pass
```

### IJobManagementService

```python
from typing import Dict, Any, Optional, List
from enum import Enum

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class IJobManagementService:
    """Job management service interface."""
    
    def create_job(self, job_type: str, parameters: Dict[str, Any], 
                   priority: JobPriority = JobPriority.NORMAL) -> str:
        """
        Create a new background job.
        
        Args:
            job_type: Type of job to create
            parameters: Job parameters
            priority: Job priority level
            
        Returns:
            Job ID string
            
        Raises:
            JobCreationError: If job creation fails
        """
        pass
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get current job status.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Current job status
            
        Raises:
            JobNotFoundError: If job ID not found
        """
        pass
    
    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job execution results.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job results dictionary or None if not available
            
        Raises:
            JobNotFoundError: If job ID not found
        """
        pass
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled successfully
            
        Raises:
            JobNotFoundError: If job ID not found
        """
        pass
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> List[Dict[str, Any]]:
        """
        List jobs with optional status filter.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of job information dictionaries
        """
        pass
```

### IPipelineConfigService

```python
from typing import Dict, Any, Optional

class IPipelineConfigService:
    """Pipeline configuration service interface."""
    
    def create_pipeline_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create standardized pipeline configuration.
        
        Args:
            config_data: Raw configuration data
            
        Returns:
            Validated pipeline configuration
            
        Raises:
            ConfigurationException: If configuration is invalid
        """
        pass
    
    def get_pipeline_config(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """
        Get saved pipeline configuration.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            Pipeline configuration or None if not found
        """
        pass
    
    def validate_pipeline_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate pipeline configuration.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationException: If configuration is invalid
        """
        pass
    
    def execute_pipeline(self, config: Dict[str, Any]) -> str:
        """
        Execute pipeline with configuration.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Execution job ID
            
        Raises:
            PipelineExecutionError: If pipeline execution fails
        """
        pass
```

## ML/AI Services

### IClassifierService

```python
from typing import Dict, Any, Optional
import numpy as np

class IClassifierService:
    """MIDI mood classification service interface."""
    
    def train_model(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a new classification model.
        
        Args:
            parameters: Training parameters including:
                - dataset: Dataset name
                - model_type: Model architecture type
                - epochs: Number of training epochs
                - batch_size: Training batch size
                - learning_rate: Learning rate
                
        Returns:
            Training results dictionary with:
                - model_path: Path to saved model
                - metrics: Training metrics
                - history: Training history
                
        Raises:
            TrainingError: If training fails
        """
        pass
    
    def classify_midi(self, midi_data: Any, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify MIDI data for mood.
        
        Args:
            midi_data: MIDI data to classify
            model_path: Optional path to model (uses default if None)
            
        Returns:
            Classification results dictionary with:
                - predicted_mood: Predicted mood category
                - probabilities: Mood probabilities
                - confidence: Prediction confidence
                
        Raises:
            ClassificationError: If classification fails
            ModelNotFoundError: If model not found
        """
        pass
    
    def evaluate_model(self, model_path: str, test_data: Any) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            model_path: Path to model file
            test_data: Test dataset
            
        Returns:
            Evaluation metrics dictionary with:
                - accuracy: Classification accuracy
                - precision: Per-class precision
                - recall: Per-class recall
                - f1_score: Per-class F1 scores
                - confusion_matrix: Confusion matrix
                
        Raises:
            EvaluationError: If evaluation fails
            ModelNotFoundError: If model not found
        """
        pass
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained model from file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if model loaded successfully
            
        Raises:
            ModelLoadError: If model loading fails
        """
        pass
```

### IEncoderService

```python
from typing import List, Any

class IEncoderService:
    """Musical data encoding service interface."""
    
    def encode_midi(self, midi_data: Any, encoding_type: str = "remi") -> List[int]:
        """
        Encode MIDI data to token sequence.
        
        Args:
            midi_data: MIDI data to encode
            encoding_type: Encoding format ("remi", "miditok", etc.)
            
        Returns:
            Token sequence as list of integers
            
        Raises:
            EncodingError: If encoding fails
        """
        pass
    
    def decode_tokens(self, tokens: List[int], encoding_type: str = "remi") -> Any:
        """
        Decode token sequence back to MIDI.
        
        Args:
            tokens: Token sequence
            encoding_type: Encoding format
            
        Returns:
            MIDI data structure
            
        Raises:
            DecodingError: If decoding fails
        """
        pass
    
    def load_vocab(self, vocab_path: str) -> bool:
        """
        Load vocabulary for encoding/decoding.
        
        Args:
            vocab_path: Path to vocabulary file
            
        Returns:
            True if vocabulary loaded successfully
            
        Raises:
            VocabularyLoadError: If vocabulary loading fails
        """
        pass
    
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Number of tokens in vocabulary
            
        Raises:
            VocabularyNotLoadedError: If vocabulary not loaded
        """
        pass
```

### IFeatureExtractionService

```python
from typing import Dict, Any
import numpy as np

class IFeatureExtractionService:
    """Feature extraction service interface."""
    
    def extract_symbolic_features(self, data: Any) -> Dict[str, Any]:
        """
        Extract symbolic music features.
        
        Args:
            data: Musical data
            
        Returns:
            Feature dictionary with:
                - pitch_features: Pitch-related features
                - rhythm_features: Rhythm-related features
                - harmony_features: Harmony-related features
                - structure_features: Structure-related features
                
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        pass
    
    def extract_latent_features(self, data: Any, model_path: str) -> np.ndarray:
        """
        Extract latent features using trained model.
        
        Args:
            data: Input data
            model_path: Path to feature extraction model
            
        Returns:
            Latent feature vector
            
        Raises:
            FeatureExtractionError: If feature extraction fails
            ModelNotFoundError: If model not found
        """
        pass
    
    def initialize_vae_model(self, config: Dict[str, Any]) -> bool:
        """
        Initialize VAE model for feature extraction.
        
        Args:
            config: VAE model configuration with:
                - model_path: Path to VAE model
                - latent_dim: Latent dimension size
                - encoder_config: Encoder configuration
                
        Returns:
            True if initialization successful
            
        Raises:
            ModelInitializationError: If initialization fails
        """
        pass
    
    def extract_all_features(self, data: Any) -> Dict[str, Any]:
        """
        Extract all available features.
        
        Args:
            data: Input data
            
        Returns:
            Comprehensive feature dictionary
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        pass
```

### IGeneratorService

```python
from typing import Dict, Any

class IGeneratorService:
    """Music generation service interface."""
    
    def generate_music(self, parameters: Dict[str, Any]) -> Any:
        """
        Generate musical content.
        
        Args:
            parameters: Generation parameters with:
                - length: Generated sequence length
                - temperature: Sampling temperature
                - top_k: Top-k sampling parameter
                - top_p: Top-p sampling parameter
                - seed: Random seed
                
        Returns:
            Generated MIDI data
            
        Raises:
            GenerationError: If generation fails
        """
        pass
    
    def initialize_model(self, config: Dict[str, Any]) -> bool:
        """
        Initialize generation model.
        
        Args:
            config: Model configuration with:
                - model_path: Path to model file
                - vocab_size: Vocabulary size
                - max_length: Maximum sequence length
                - device: Computation device
                
        Returns:
            True if initialization successful
            
        Raises:
            ModelInitializationError: If initialization fails
        """
        pass
    
    def set_generation_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Set generation parameters.
        
        Args:
            parameters: Generation parameters
            
        Returns:
            True if parameters set successfully
            
        Raises:
            ParameterError: If parameters are invalid
        """
        pass
    
    def generate_with_conditioning(self, conditioning_data: Any, 
                                  parameters: Dict[str, Any]) -> Any:
        """
        Generate music with conditioning.
        
        Args:
            conditioning_data: Conditioning information
            parameters: Generation parameters
            
        Returns:
            Conditioned generated music
            
        Raises:
            GenerationError: If generation fails
        """
        pass
```

### IMultimodalMappingService

```python
from typing import Dict, Any
import numpy as np

class IMultimodalMappingService:
    """Multimodal mapping service interface."""
    
    def map_text_to_music(self, text: str, parameters: Dict[str, Any]) -> Any:
        """
        Map text descriptions to musical features.
        
        Args:
            text: Text description
            parameters: Mapping parameters with:
                - style: Musical style
                - mood: Target mood
                - instrumentation: Instrumentation preferences
                
        Returns:
            Musical representation
            
        Raises:
            MappingError: If mapping fails
        """
        pass
    
    def map_image_to_music(self, image_data: Any, parameters: Dict[str, Any]) -> Any:
        """
        Map image data to musical features.
        
        Args:
            image_data: Image data
            parameters: Mapping parameters
            
        Returns:
            Musical representation
            
        Raises:
            MappingError: If mapping fails
        """
        pass
    
    def initialize_model(self, config: Dict[str, Any]) -> bool:
        """
        Initialize multimodal mapping model.
        
        Args:
            config: Model configuration with:
                - text_encoder: Text encoder type
                - image_encoder: Image encoder type
                - embedding_dim: Embedding dimension
                - fusion_method: Fusion method
                
        Returns:
            True if initialization successful
            
        Raises:
            ModelInitializationError: If initialization fails
        """
        pass
    
    def encode_modality(self, data: Any, modality: str) -> np.ndarray:
        """
        Encode data from specific modality.
        
        Args:
            data: Input data
            modality: Modality type ("text", "image", "audio")
            
        Returns:
            Encoded representation
            
        Raises:
            EncodingError: If encoding fails
        """
        pass
```

## Utility Services

### IMusicBaseService

```python
from typing import List, Dict, Any

class IMusicBaseService:
    """Basic music processing service interface."""
    
    def extract_chord_progression(self, midi_data: Any) -> List[str]:
        """
        Extract chord progression from MIDI.
        
        Args:
            midi_data: MIDI data
            
        Returns:
            List of chord symbols
            
        Raises:
            ChordExtractionError: If extraction fails
        """
        pass
    
    def analyze_rhythm(self, midi_data: Any) -> Dict[str, Any]:
        """
        Analyze rhythmic patterns.
        
        Args:
            midi_data: MIDI data
            
        Returns:
            Rhythm analysis with:
                - tempo: Detected tempo
                - time_signature: Time signature
                - rhythm_patterns: Detected patterns
                - syncopation: Syncopation measure
                
        Raises:
            RhythmAnalysisError: If analysis fails
        """
        pass
    
    def transpose_midi(self, midi_data: Any, semitones: int) -> Any:
        """
        Transpose MIDI data.
        
        Args:
            midi_data: MIDI data
            semitones: Number of semitones to transpose
            
        Returns:
            Transposed MIDI data
            
        Raises:
            TranspositionError: If transposition fails
        """
        pass
    
    def get_midi_statistics(self, midi_data: Any) -> Dict[str, Any]:
        """
        Compute basic MIDI statistics.
        
        Args:
            midi_data: MIDI data
            
        Returns:
            Statistics dictionary with:
                - duration: Total duration
                - note_count: Number of notes
                - pitch_range: Pitch range
                - velocity_stats: Velocity statistics
                
        Raises:
            StatisticsError: If computation fails
        """
        pass
```

### IDataloaderService

```python
from typing import Dict, Any

class IDataloaderService:
    """Data loading service interface."""
    
    def create_dataloader(self, config: Dict[str, Any]) -> Any:
        """
        Create data loader instance.
        
        Args:
            config: Data loader configuration with:
                - dataset: Dataset name
                - batch_size: Batch size
                - shuffle: Whether to shuffle data
                - num_workers: Number of worker processes
                - pin_memory: Whether to pin memory
                
        Returns:
            Configured data loader
            
        Raises:
            DataloaderCreationError: If creation fails
        """
        pass
    
    def load_dataset(self, dataset_name: str, split: str = "train") -> Any:
        """
        Load specified dataset.
        
        Args:
            dataset_name: Name of dataset
            split: Dataset split ("train", "val", "test")
            
        Returns:
            Dataset object
            
        Raises:
            DatasetNotFoundError: If dataset not found
        """
        pass
    
    def preprocess_data(self, data: Any, preprocessing_config: Dict[str, Any]) -> Any:
        """
        Preprocess data according to configuration.
        
        Args:
            data: Input data
            preprocessing_config: Preprocessing configuration
            
        Returns:
            Preprocessed data
            
        Raises:
            PreprocessingError: If preprocessing fails
        """
        pass
    
    def get_data_statistics(self, dataset: Any) -> Dict[str, Any]:
        """
        Compute dataset statistics.
        
        Args:
            dataset: Dataset object
            
        Returns:
            Statistics dictionary with:
                - size: Dataset size
                - feature_stats: Feature statistics
                - label_distribution: Label distribution
                
        Raises:
            StatisticsError: If computation fails
        """
        pass
```

### IPseudoLabellerService

```python
from typing import Dict, Any

class IPseudoLabellerService:
    """Pseudo-labeling service interface."""
    
    def generate_pseudo_labels(self, data: Any, model_path: str, 
                              confidence_threshold: float = 0.9) -> Dict[str, Any]:
        """
        Generate pseudo labels for unlabeled data.
        
        Args:
            data: Unlabeled data
            model_path: Path to labeling model
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Pseudo-labeled data with:
                - labels: Generated labels
                - confidences: Label confidences
                - filtered_data: High-confidence data
                
        Raises:
            PseudoLabelingError: If labeling fails
        """
        pass
    
    def initialize_model(self, config: Dict[str, Any]) -> bool:
        """
        Initialize pseudo-labeling model.
        
        Args:
            config: Model configuration with:
                - model_path: Path to model
                - threshold: Confidence threshold
                - batch_size: Processing batch size
                
        Returns:
            True if initialization successful
            
        Raises:
            ModelInitializationError: If initialization fails
        """
        pass
    
    def filter_by_confidence(self, pseudo_labels: Dict[str, Any], 
                            threshold: float) -> Dict[str, Any]:
        """
        Filter pseudo labels by confidence.
        
        Args:
            pseudo_labels: Pseudo-labeled data
            threshold: Confidence threshold
            
        Returns:
            Filtered pseudo-labeled data
            
        Raises:
            FilteringError: If filtering fails
        """
        pass
    
    def evaluate_pseudo_labels(self, pseudo_labels: Dict[str, Any], 
                              ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate pseudo label quality.
        
        Args:
            pseudo_labels: Pseudo-labeled data
            ground_truth: Ground truth labels
            
        Returns:
            Evaluation metrics with:
                - accuracy: Pseudo-label accuracy
                - precision: Precision per class
                - recall: Recall per class
                - coverage: Data coverage
                
        Raises:
            EvaluationError: If evaluation fails
        """
        pass
```

## Exception Classes

### Core Exceptions

```python
class ServiceException(Exception):
    """Base exception for all service errors."""
    pass

class ConfigurationException(ServiceException):
    """Configuration-related errors."""
    pass

class ModelNotFoundError(ServiceException):
    """Model file not found errors."""
    pass

class ModelInitializationError(ServiceException):
    """Model initialization errors."""
    pass

class ModelLoadError(ServiceException):
    """Model loading errors."""
    pass
```

### ML/AI Exceptions

```python
class TrainingError(ServiceException):
    """Training-related errors."""
    pass

class ClassificationError(ServiceException):
    """Classification errors."""
    pass

class EvaluationError(ServiceException):
    """Model evaluation errors."""
    pass

class EncodingError(ServiceException):
    """Data encoding errors."""
    pass

class DecodingError(ServiceException):
    """Data decoding errors."""
    pass

class GenerationError(ServiceException):
    """Music generation errors."""
    pass

class FeatureExtractionError(ServiceException):
    """Feature extraction errors."""
    pass

class MappingError(ServiceException):
    """Multimodal mapping errors."""
    pass
```

### Job Management Exceptions

```python
class JobException(ServiceException):
    """Base job management exception."""
    pass

class JobCreationError(JobException):
    """Job creation errors."""
    pass

class JobNotFoundError(JobException):
    """Job not found errors."""
    pass

class PipelineExecutionError(ServiceException):
    """Pipeline execution errors."""
    pass
```

### Data Processing Exceptions

```python
class DataException(ServiceException):
    """Base data processing exception."""
    pass

class DataloaderCreationError(DataException):
    """Data loader creation errors."""
    pass

class DatasetNotFoundError(DataException):
    """Dataset not found errors."""
    pass

class PreprocessingError(DataException):
    """Data preprocessing errors."""
    pass

class StatisticsError(DataException):
    """Statistics computation errors."""
    pass
```

### Music Processing Exceptions

```python
class MusicProcessingError(ServiceException):
    """Base music processing exception."""
    pass

class ChordExtractionError(MusicProcessingError):
    """Chord extraction errors."""
    pass

class RhythmAnalysisError(MusicProcessingError):
    """Rhythm analysis errors."""
    pass

class TranspositionError(MusicProcessingError):
    """Transposition errors."""
    pass
```

## Common Patterns

### Service Resolution Pattern

```python
from domain.interfaces.dependency_injection import get_dependency_container

def resolve_service(service_interface):
    """Helper function to resolve services."""
    container = get_dependency_container()
    return container.resolve(service_interface)
```

### Error Handling Pattern

```python
def safe_service_call(service_method, *args, **kwargs):
    """Safe service call with error handling."""
    try:
        return service_method(*args, **kwargs)
    except ServiceException as e:
        logger.error(f"Service error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise ServiceException(f"Unexpected error: {e}")
```

### Async Service Pattern

```python
import asyncio

async def async_service_call(service, method_name, *args, **kwargs):
    """Async wrapper for service calls."""
    loop = asyncio.get_event_loop()
    method = getattr(service, method_name)
    return await loop.run_in_executor(None, method, *args, **kwargs)
```

## Type Definitions

### Common Types

```python
from typing import TypedDict, Union, List, Dict, Any
import numpy as np

# MIDI Data Types
MIDIData = Union[str, bytes, Any]  # Path, binary data, or MIDI object
TokenSequence = List[int]
FeatureVector = np.ndarray

# Configuration Types
ServiceConfig = Dict[str, Any]
PipelineConfig = Dict[str, Any]
ModelConfig = Dict[str, Any]

# Result Types
class ClassificationResult(TypedDict):
    predicted_mood: str
    probabilities: Dict[str, float]
    confidence: float

class TrainingResult(TypedDict):
    model_path: str
    metrics: Dict[str, float]
    history: Dict[str, List[float]]

class EvaluationResult(TypedDict):
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: List[List[int]]
```

## Version Information

- **API Version**: 1.0.0
- **Last Updated**: December 2024
- **Compatibility**: Python 3.9+
- **Dependencies**: See `requirements.txt`

For implementation details, see the [Service Interface Documentation](SERVICE_INTERFACES.md). 