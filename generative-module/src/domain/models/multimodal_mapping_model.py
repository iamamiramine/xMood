from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class MultimodalMappingParameters(BaseModel):
    """
    Parameters for multimodal mapping service.
    
    Enhanced to handle the data formats from ReMIDICaps dataset.
    """
    # Output configuration
    output_folder: Optional[str] = Field(None, description="Folder to save generated features")
    output_name: str = Field("multimodal_features", description="Name of the output file")
    
    # Model configuration
    weights_path: str = Field(..., description="Path to the model weights")
    config_path: str = Field(..., description="Path to the model configuration file")
    
    # Generation parameters
    num_bars: int = Field(8, description="Number of bars to generate")
    temperature: float = Field(1.0, description="Temperature for sampling during generation") 