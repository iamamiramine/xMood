from typing import Optional, List
from pydantic import BaseModel, Field


class PseudoLabellerParameters(BaseModel):
    """Parameters for creating mood labels for a directory of images."""
    
    # Input configuration
    input_dir: str = Field(..., description="Path to the directory containing images to be labeled")
    
    # Output configuration
    output_file: str = Field("image_mood_labels.json", description="Path to the output JSON file where labels will be saved")
    
    # Processing configuration
    device: str = Field("cuda", description="Device to run the CLIP model on (cuda, cpu)")
    
    # Model configuration
    model_name: str = Field("ViT-B/32", description="CLIP model name to use")
    
    # Processing options
    batch_size: int = Field(1, description="Batch size for processing images")
    valid_extensions: List[str] = Field(
        default_factory=lambda: ['.png', '.jpg', '.jpeg', '.bmp', '.gif'],
        description="Valid image file extensions"
    )
    
    # Mood categories
    mood_categories: List[str] = Field(
        default_factory=lambda: [
            'relaxing', 'christmas', 'dramatic', 'meditative',
            'energetic', 'happy', 'motivational', 'dark', 'love'
        ],
        description="Mood categories for classification"
    )


class ImageMoodClassificationParameters(BaseModel):
    """Parameters for classifying a single image's mood."""
    
    # Input configuration
    image_path: str = Field(..., description="Path to the image file to classify")
    
    # Model configuration
    device: str = Field("cuda", description="Device to run the CLIP model on (cuda, cpu)")
    model_name: str = Field("ViT-B/32", description="CLIP model name to use")
    
    # Mood categories
    mood_categories: List[str] = Field(
        default_factory=lambda: [
            'relaxing', 'christmas', 'dramatic', 'meditative',
            'energetic', 'happy', 'motivational', 'dark', 'love'
        ],
        description="Mood categories for classification"
    )
    
    # Output configuration
    return_probabilities: bool = Field(True, description="Whether to return probability scores")
    return_top_k: int = Field(3, description="Number of top predictions to return")


class BatchImageMoodClassificationParameters(BaseModel):
    """Parameters for batch mood classification of multiple images."""
    
    # Input configuration
    input_dir: str = Field(..., description="Directory containing images to classify")
    image_paths: Optional[List[str]] = Field(None, description="Specific image paths to process")
    
    # Processing configuration
    batch_size: int = Field(8, description="Batch size for processing")
    num_workers: int = Field(2, description="Number of workers for data loading")
    device: str = Field("cuda", description="Device to run the CLIP model on")
    
    # Model configuration
    model_name: str = Field("ViT-B/32", description="CLIP model name to use")
    
    # Mood categories
    mood_categories: List[str] = Field(
        default_factory=lambda: [
            'relaxing', 'christmas', 'dramatic', 'meditative',
            'energetic', 'happy', 'motivational', 'dark', 'love'
        ],
        description="Mood categories for classification"
    )
    
    # File filtering
    valid_extensions: List[str] = Field(
        default_factory=lambda: ['.png', '.jpg', '.jpeg', '.bmp', '.gif'],
        description="Valid image file extensions"
    )
    
    # Output configuration
    output_file: Optional[str] = Field(None, description="Path to save results JSON file")
    save_individual: bool = Field(False, description="Whether to save individual predictions")
    return_probabilities: bool = Field(True, description="Whether to return probability scores") 