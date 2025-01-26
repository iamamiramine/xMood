from typing import List, Optional
from pydantic import BaseModel, Field


class CaptionParameters(BaseModel):
    prompt: str = Field(description="The prompt to generate caption from")
    model_id: str = Field(default="bigscience/bloom-1b7", description="HuggingFace model ID")
    max_new_tokens: int = Field(default=1000, description="Maximum number of tokens to generate")
    chain_type: str = Field(default="basic", description="Type of chain to use (basic or rag)")


class CaptionDatasetParameters(BaseModel):
    model_id: str = Field(default="bigscience/bloom-1b7", description="HuggingFace model ID")
    max_new_tokens: int = Field(default=1000, description="Maximum number of tokens to generate")
    chain_type: str = Field(default="basic", description="Type of chain to use (basic or rag)")
    force_reload: bool = Field(default=False, description="Force reload of document chunks")
    batch_size: int = Field(default=4, description="Number of files to process in parallel")
