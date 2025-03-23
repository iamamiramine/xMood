import torch
import json
from transformers.models.bert.modeling_bert import BertAttention

from application.generator.models.generator_model import MIDIGeneratorModule


def load_generator_from_checkpoint(checkpoint_path: str, eval=True):
    pl_ckpt = torch.load(checkpoint_path, map_location="cpu")
    kwargs = pl_ckpt["hyper_parameters"]
    model = MIDIGeneratorModule(**kwargs)
    state_dict = pl_ckpt["state_dict"]
    # position_ids are no longer saved in the state_dict starting with transformers==4.31.0
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith("embeddings.position_ids")}
    try:
        # succeeds for checkpoints trained with transformers>4.13.0
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        config = model.transformer.decoder.bert.config
        for layer in model.transformer.decoder.bert.encoder.layer:
            layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
        model.load_state_dict(state_dict)
    if eval:
        model.freeze()
        model.eval()
    return model


def load_from_checkpoint_new(weights_path: str, config_path: str, eval=True):
    """
    Load a generator model from separate weights (.pt) and config (.json) files

    Args:
        weights_path: Path to the weights file (.pt)
        config_path: Path to the hyperparameters config file (.json)
        eval: Whether to freeze and set the model to eval mode

    Returns:
        Loaded model
    """
    # Load hyperparameters from JSON
    with open(config_path, "r") as f:
        kwargs = json.load(f)

    # Create model with the hyperparameters
    model = MIDIGeneratorModule(**kwargs)

    # Load weights
    state_dict = torch.load(weights_path, map_location="cpu")

    try:
        # succeeds for models trained with transformers>4.13.0
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Fallback for older models (same as in original function)
        config = model.transformer.decoder.bert.config
        for layer in model.transformer.decoder.bert.encoder.layer:
            layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
        model.load_state_dict(state_dict)

    if eval:
        model.freeze()
        model.eval()

    return model
