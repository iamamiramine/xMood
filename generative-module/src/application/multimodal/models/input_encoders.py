import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig

class TextEncoder(nn.Module):
    """Text symbolic using DistilBERT to process emotional text inputs."""
    
    def __init__(self, output_dim=384, pretrained=True):
        super(TextEncoder, self).__init__()
        
        # Initialize DistilBERT model
        if pretrained:
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        else:
            config = DistilBertConfig()
            self.bert = DistilBertModel(config)
            
        # Projection layer to get the desired output dimension
        self.projection = nn.Linear(self.bert.config.hidden_size, output_dim)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the text symbolic.
        
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor indicating which tokens to attend to
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        # Get DistilBERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Project to the desired output dimension
        return self.projection(cls_output)


class StructuredFeatureProcessor(nn.Module):
    """
    Processes structured features like mood scores and music parameters.
    Enhanced to better handle tokenized/categorical nature of inputs seen in the dataset.
    """
    
    def __init__(self, input_dim, output_dim=256, d_model=512, dropout=0.1):
        super(StructuredFeatureProcessor, self).__init__()
        
        # More robust MLP with layer normalization and dropout
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, features):
        """
        Forward pass of the structured feature processor.
        
        Args:
            features: Tensor of structured features - could be:
                     - Mood vectors (e.g., from "Mood_Relaxing_0.2844 Mood_Christmas_0.2736...")
                     - Global features (e.g., time signature, key, instruments, chords...)
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        # Handle potential NaNs in the input
        features = torch.nan_to_num(features, nan=0.0)
        return self.mlp(features)


class GlobalFeatureProcessor(StructuredFeatureProcessor):
    """
    Specialization of StructuredFeatureProcessor for global features.
    Designed to handle the specific structure of global features in the dataset.
    
    Global features include:
    - Time signature (e.g., "Time Signature_4/4")
    - Key signature (e.g., "Key Signature_D:maj")
    - Note density, mean velocity, mean pitch, mean duration
    - Instruments and chords
    """
    
    def __init__(self, input_dim, output_dim=256, d_model=512, dropout=0.1):
        super(GlobalFeatureProcessor, self).__init__(
            input_dim=input_dim, 
            output_dim=output_dim, 
            d_model=d_model,
            dropout=dropout
        )


class MoodProcessor(StructuredFeatureProcessor):
    """
    Specialization of StructuredFeatureProcessor for mood features.
    Designed to handle the specific structure of mood values in the dataset.
    
    Mood features are typically in format:
    "Mood_Relaxing_0.2844 Mood_Christmas_0.2736 Mood_Dramatic_0.2423 Mood_Meditative_0.1997"
    """
    
    def __init__(self, input_dim, output_dim=256, d_model=512, dropout=0.1):
        super(MoodProcessor, self).__init__(
            input_dim=input_dim, 
            output_dim=output_dim, 
            d_model=d_model,
            dropout=dropout
        ) 