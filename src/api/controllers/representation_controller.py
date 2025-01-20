from fastapi import APIRouter

from src.application.representation.services import representation_services
from src.domain.models.representation.representation_model import (
    RepresentationParameters,
)

router = APIRouter()


@router.post("/represent_encoding")
def represent_encoding(parameters: RepresentationParameters) -> dict:
    """
    Description:
    ------------
        Representation of Encoding

    Parameters:
    -----------
        file,
        dataset_name,
        context_size,
        max_bars,
        max_positions,
        bar_token_mask,
        max_bars_per_context,
        max_contexts_per_file,
        load_symb
        load_latent,

    Returns:
    --------
    dict
        A dictionary

    """
    return representation_services.run_representation(parameters)
