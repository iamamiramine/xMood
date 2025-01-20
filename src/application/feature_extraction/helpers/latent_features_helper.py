import torch

from transformers.models.bert.modeling_bert import BertAttention

from src.application.feature_extraction.models.vae_model import VqVaeModule


def load_vae_from_checkpoint(checkpoint_dir: str):
    pl_ckpt = torch.load(checkpoint_dir, map_location="cpu")
    kwargs = pl_ckpt["hyper_parameters"]
    model = VqVaeModule(**kwargs)
    state_dict = pl_ckpt["state_dict"]
    # position_ids are no longer saved in the state_dict starting with transformers==4.31.0
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith("embeddings.position_ids")}
    try:
        # succeeds for checkpoints trained with transformers>4.13.0
        model.load_state_dict(state_dict)
    except RuntimeError:
        # work around a breaking change introduced in transformers==4.13.0, which fixed the position_embedding_type of cross-attention modules "absolute"
        config = model.transformer.decoder.bert.config
        for layer in model.transformer.decoder.bert.encoder.layer:
            layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
        model.load_state_dict(state_dict)
    model.freeze()
    model.eval()
    model.cpu()
    return model
