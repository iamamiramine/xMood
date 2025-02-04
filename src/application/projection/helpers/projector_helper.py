import torch
from transformers.models.bert.modeling_bert import BertAttention

from application.projection.models.projection_model import ProjectorModule


def load_projector_from_checkpoint(checkpoint_dir: str, eval=True):
    pl_ckpt = torch.load(checkpoint_dir, map_location="cpu")
    kwargs = pl_ckpt["hyper_parameters"]
    model = ProjectorModule(**kwargs)
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
