from fastapi import Query

from typing import Annotated, Tuple, Optional

from src.domain.models.base_model import BaseEnum
from pydantic import BaseModel


class GeneratorTrainingParameters(BaseModel):
    dataset_name: Annotated[str, Query(description="Name of Dataset")]
    training_name: Annotated[str, Query(description="Output Directory")] = ""

    load_latent: Annotated[bool, Query(description="Latent")] = False
    load_desc: Annotated[bool, Query(description="Description")] = False
    load_sentiments: Annotated[bool, Query(description="Sentiments")] = False

    load_from_checkpoint: Annotated[bool, Query(description="Load from checkpoint")] = False
    checkpoint_name: Annotated[str, Query(description="Checkpoint")] = ""

    load_bert_from_ckpt: Annotated[bool, Query(description="Load BERT from checkpoint")] = False

    device: Annotated[str, Query(description="Device")] = "cuda"

    epochs: Annotated[int, Query(description="Number of epochs")] = 100
    max_training_steps: Annotated[int, Query(description="Max. number of training iterations")] = 100000
    target_batch_size: Annotated[
        int, Query(description="Number of samples in each backward step, gradients will be accumulated over TARGET_BATCH_SIZE//BATCH_SIZE batches")
    ] = 256

    lr: Annotated[float, Query(description="Learning rate")] = 1e-4
    lr_schedule: Annotated[str, Query(description="Learning rate schedule")] = "sqrt_decay"
    warmup_steps: Annotated[Optional[int], Query(description="Warmup steps")] = 4000
    max_steps: Annotated[Optional[int], Query(description="Maximum steps")] = 100000000000000000000
    encoder_layers: Annotated[int, Query(description="Number of encoder layers")] = 6
    decoder_layers: Annotated[int, Query(description="Number of decoder layers")] = 12
    intermediate_size: Annotated[int, Query(description="Intermediate Layers Size")] = 2048
    num_attention_heads: Annotated[int, Query(description="Number of attention heads")] = 8
    use_pretrained_latent_embeddings: Annotated[bool, Query(description="Use pretrained latent embeddings")] = True


class GeneratorGenerateParameters(BaseModel):
    dataset_name: Annotated[str, Query(description="Name of Dataset")]

    load_latent: Annotated[bool, Query(description="Latent")] = True
    load_desc: Annotated[bool, Query(description="Description")] = False

    training_name: Annotated[str, Query(description="Training Name")] = ""
    load_from_checkpoint: Annotated[bool, Query(description="Load from checkpoint")] = False
    checkpoint_name: Annotated[str, Query(description="Checkpoint")] = ""

    max_bars: Annotated[int, Query(description="Maximum number of bars")] = 512
    make_medleys: Annotated[bool, Query(description="Make medleys")] = False
    n_medley_pieces: Annotated[int, Query(description="Number of medley pieces")] = 2
    n_medley_bars: Annotated[int, Query(description="Number of medley bars")] = 16

    max_iter: Annotated[int, Query(description="Maximum iterations")] = 16_000
    max_n_files: Annotated[int, Query(description="Maximum number of files")] = -1

    verbose: Annotated[bool, Query(description="Verbose")] = False


class GeneratorGeneratePromptParameters(BaseModel):
    dataset_name: Annotated[str, Query(description="Name of Dataset")]
    prompt_name: Annotated[str, Query(description="Prompt Name")]

    label_names: Annotated[list[str], Query(description="Label Names")] = []
    label_scores: Annotated[list[float], Query(description="Label Scores")] = []

    training_name: Annotated[str, Query(description="Training Name")] = ""
    load_from_checkpoint: Annotated[bool, Query(description="Load from checkpoint")] = False
    checkpoint_name: Annotated[str, Query(description="Checkpoint")] = ""

    max_bars: Annotated[int, Query(description="Maximum number of bars")] = 512
    make_medleys: Annotated[bool, Query(description="Make medleys")] = False
    n_medley_pieces: Annotated[int, Query(description="Number of medley pieces")] = 2
    n_medley_bars: Annotated[int, Query(description="Number of medley bars")] = 16

    max_n_tokens: Annotated[int, Query(description="Maximum number of tokens")] = 2048
    initial_context: Annotated[int, Query(description="Initial Number of Tokens")] = 1

    verbose: Annotated[bool, Query(description="Verbose")] = False
