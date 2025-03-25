import torch
from transformers import DistilBertTokenizer

import json
from application.multimodal_mapping.models.input_encoders import TextEncoder, StructuredFeatureProcessor, GlobalFeatureProcessor, MoodProcessor
from application.multimodal_mapping.models.embedding_fusion import CrossAttentionFusion, LinearConcatFusion
from application.multimodal_mapping.models.hierarchical_vae import TransformerVAE


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


def load_mapping_from_checkpoint(weights_path, config_path, eval=True):
    """
    Load a multimodal mapping model from checkpoint files.

    Args:
        weights_path: Path to the weights file (.pt)
        config_path: Path to the config file (.json)
        eval: Whether to set the model to evaluation mode

    Returns:
        dict: Dictionary containing model components
    """

    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load state dict
    state_dict = torch.load(weights_path, map_location="cpu")

    # Initialize model components
    text_encoder = TextEncoder(output_dim=config.get("text_dim", 384))

    # Initialize specialized feature processors for global features and moods
    global_feature_dim = config.get("global_feature_dim", 64)
    global_feature_out_dim = config.get("global_feature_out_dim", 128)
    global_feature_processor = GlobalFeatureProcessor(
        input_dim=global_feature_dim, output_dim=global_feature_out_dim, hidden_dim=config.get("hidden_dim", 512), dropout=config.get("dropout", 0.1)
    )

    mood_dim = config.get("mood_dim", 32)
    mood_out_dim = config.get("mood_out_dim", 128)
    mood_processor = MoodProcessor(input_dim=mood_dim, output_dim=mood_out_dim, hidden_dim=config.get("hidden_dim", 512), dropout=config.get("dropout", 0.1))

    # Embedding fusion
    fusion_type = config.get("fusion_type", "linear_concat")
    if fusion_type == "cross_attention":
        fusion_module = CrossAttentionFusion(embed_dim=config.get("fusion_dim", 512), num_heads=config.get("fusion_heads", 8))
    else:
        fusion_module = LinearConcatFusion(
            input_dims=[config.get("text_dim", 384), global_feature_out_dim, mood_out_dim],
            output_dim=config.get("fusion_dim", 512),
        )

    # Transformer VAE (renamed from hierarchical)
    vae_model = TransformerVAE(
        input_dim=config.get("fusion_dim", 512),
        output_dim=config.get("output_dim", 256),
        hidden_dim=config.get("hidden_dim", 512),
        latent_dim=config.get("latent_dim", 128),
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 8),
        context_size=config.get("context_size", 512),
        dropout=config.get("dropout", 0.1),
    )

    # Load state dicts
    text_encoder.load_state_dict(state_dict["text_encoder"])

    # Handle different state dict variations
    if "global_feature_processor" in state_dict:
        global_feature_processor.load_state_dict(state_dict["global_feature_processor"])
    else:
        # For backward compatibility
        print("Using generic processor state_dict for specialized GlobalFeatureProcessor")
        global_feature_processor.load_state_dict(state_dict["structured_feature_processor"])

    if "mood_processor" in state_dict:
        mood_processor.load_state_dict(state_dict["mood_processor"])
    else:
        # For backward compatibility
        print("Using generic processor state_dict for specialized MoodProcessor")
        mood_processor.load_state_dict(state_dict["structured_feature_processor"])

    fusion_module.load_state_dict(state_dict["fusion_module"])

    # Handle different VAE state dict names
    if "vae" in state_dict:
        vae_model.load_state_dict(state_dict["vae"])
    elif "hierarchical_vae" in state_dict:
        vae_model.load_state_dict(state_dict["hierarchical_vae"])
    else:
        raise ValueError("Could not find VAE model weights in state_dict")

    # Set to evaluation mode if needed
    if eval:
        text_encoder.eval()
        global_feature_processor.eval()
        mood_processor.eval()
        fusion_module.eval()
        vae_model.eval()

    return {
        "text_encoder": text_encoder,
        "global_feature_processor": global_feature_processor,
        "mood_processor": mood_processor,
        "fusion_module": fusion_module,
        "vae": vae_model,  # Changed key from hierarchical_vae to vae
        "config": config,
    }
