import torch
from transformers import DistilBertTokenizer

import json
from application.multimodal_mapping.models.input_encoders import TextEncoder, StructuredFeatureProcessor, GlobalFeatureProcessor, MoodProcessor
from application.multimodal_mapping.models.embedding_fusion import CrossAttentionFusion, LinearConcatFusion
from application.multimodal_mapping.models.multimodal_mapping import MultimodalMappingModule
from application.encoder.models.vocab_model import SymbolicFeaturesVocab, MoodsVocab


def initialize_tokenizers_and_processors():
    """
    Initialize tokenizers and processors for text inputs.

    Returns:
        tuple: (text_tokenizer, None)
    """

    # Text tokenizer
    text_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Return only text tokenizer, with None for image processor
    return text_tokenizer, None


def load_from_checkpoint(checkpoint_path: str, eval=True):
    """
    Load a multimodal mapping model from a PyTorch Lightning checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file (.ckpt)
        eval: Whether to set the model to evaluation mode

    Returns:
        Loaded model
    """
    # Load the PyTorch Lightning checkpoint
    pl_ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract hyperparameters from the checkpoint
    kwargs = pl_ckpt["hyper_parameters"]
    
    # Create model with the hyperparameters
    model = MultimodalMappingModule(**kwargs)
    
    # Load state dict
    model.load_state_dict(pl_ckpt["state_dict"])
    
    # Set to evaluation mode if needed
    if eval:
        model.freeze()
        model.eval()
        
    return model


def load_from_checkpoint_new(weights_path: str, config_path: str, eval=True):
    """
    Load a multimodal mapping model from separate weights (.pt) and config (.json) files

    Args:
        weights_path: Path to the weights file (.pt)
        config_path: Path to the config file (.json)
        eval: Whether to set the model to evaluation mode

    Returns:
        Loaded model
    """
    # Load hyperparameters from JSON
    with open(config_path, "r") as f:
        kwargs = json.load(f)
        
    # Create model with the hyperparameters
    model = MultimodalMappingModule(**kwargs)
    
    # Load weights
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    # Set to evaluation mode if needed
    if eval:
        model.freeze()
        model.eval()
        
    return model


def preprocess_mood_tokens(mood_string):
    """
    Convert space-separated mood tokens to numeric features.
    
    Args:
        mood_string: String like "Mood_Relaxing_0.2844 Mood_Christmas_0.2736 ..."
        
    Returns:
        torch.Tensor: Tensor of mood features
    """
    # Initialize the mood vocabulary
    mood_vocab = MoodsVocab()
    
    # Create a zero tensor with the size of all possible moods
    # Use a fixed size for all mood types in the MoodsVocab
    mood_features = torch.zeros(9)  # 9 mood categories
    
    # Map each mood token to its intensity
    mood_tokens = mood_string.split()
    for token in mood_tokens:
        parts = token.split('_')
        if len(parts) == 3:
            mood_type = parts[1].lower()  # Convert to lowercase for matching
            intensity = float(parts[2])
            
            # Find the index of this mood type
            if mood_type == 'relaxing':
                mood_features[0] = intensity
            elif mood_type == 'christmas':
                mood_features[1] = intensity
            elif mood_type == 'dramatic':
                mood_features[2] = intensity
            elif mood_type == 'meditative':
                mood_features[3] = intensity
            elif mood_type == 'energetic':
                mood_features[4] = intensity
            elif mood_type == 'happy':
                mood_features[5] = intensity
            elif mood_type == 'motivational':
                mood_features[6] = intensity
            elif mood_type == 'dark':
                mood_features[7] = intensity
            elif mood_type == 'love':
                mood_features[8] = intensity
    
    return mood_features


def preprocess_global_features(global_features_string):
    """
    Convert space-separated global feature tokens to numeric features.
    
    Args:
        global_features_string: String like "Time Signature_4/4 Key Signature_D:maj ..."
        
    Returns:
        torch.Tensor: Tensor of global features
    """
    # Initialize the symbolic features vocabulary
    symbolic_vocab = SymbolicFeaturesVocab()
    
    # Create a zero tensor with the size matching the symbolic vocabulary
    features_dim = len(symbolic_vocab)
    global_features = torch.zeros(features_dim)
    
    # Split the global features string into tokens
    feature_tokens = global_features_string.split()
    
    # Set the features based on the presence of tokens
    for token in feature_tokens:
        # Get the index of the token in the vocabulary
        token_idx = symbolic_vocab.to_i(token)
        
        # If token is in vocabulary, set its feature to 1.0
        if token_idx != symbolic_vocab.default_index:
            global_features[token_idx] = 1.0
    
    return global_features
