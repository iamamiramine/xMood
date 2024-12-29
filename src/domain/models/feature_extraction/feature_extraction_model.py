from fastapi import Query

from typing import Annotated, Tuple, Optional

from src.domain.models.base_model import BaseEnum
from pydantic import BaseModel


class DescriptionParameters(BaseModel):
    midi: Annotated[str, Query(description="File path of the encoded MIDI")] = ""
    chords_out_dir: Annotated[str, Query(description="Chords output directory")] = ""
    keys_out_dir: Annotated[str, Query(description="Keys output directory")] = ""
    save: Annotated[bool, Query(description="Save")] = False
    description_out_dir: Annotated[
        str, Query(description="Description output directory")
    ] = ""


class DescriptionDatasetParameters(BaseModel):
    dataset_name: Annotated[str, Query(description="Name of Dataset")]


class LatentRepresentationParameters(BaseModel):
    file: Annotated[str, Query(description="File path of the encoded MIDI")] = ""
    out_dir: Annotated[str, Query(description="Output directory")] = ""
    save_latents: Annotated[bool, Query(description="Save latents")] = False
    checkpoint_dir: Annotated[str, Query(description="Checkpoint")] = ""
    encoding_dir: Annotated[str, Query(description="Encoding directory")] = ""


class LatentRepresentationDatasetParameters(BaseModel):
    dataset_name: Annotated[str, Query(description="Name of Dataset")]
    out_dir: Annotated[str, Query(description="Output directory")] = ""
    training_name: Annotated[str, Query(description="Output Directory")] = ""
    checkpoint_name: Annotated[str, Query(description="Checkpoint")] = ""

    epochs: Annotated[int, Query(description="Number of epochs")] = 100
    max_training_steps: Annotated[
        int, Query(description="Max. number of training iterations")
    ] = 100000

    device: Annotated[str, Query(description="Device")] = "cuda"
    target_batch_size: Annotated[
        int,
        Query(
            description="Number of samples in each backward step, gradients will be accumulated over TARGET_BATCH_SIZE//BATCH_SIZE batches"
        ),
    ] = 256


class VQVAEParameters(BaseModel):
    dataset_name: Annotated[str, Query(description="Name of Dataset")]
    training_name: Annotated[str, Query(description="Output Directory")] = ""

    load_from_checkpoint: Annotated[bool, Query(description="Load from checkpoint")] = (
        False
    )
    checkpoint_dir: Annotated[str, Query(description="Checkpoint")] = ""
    datamodule_state: Annotated[int, Query(description="Datamodule State")] = 1

    device: Annotated[str, Query(description="Device")] = "cuda"

    epochs: Annotated[int, Query(description="Number of epochs")] = 100
    max_training_steps: Annotated[
        int, Query(description="Max. number of training iterations")
    ] = 100000
    target_batch_size: Annotated[
        int,
        Query(
            description="Number of samples in each backward step, gradients will be accumulated over TARGET_BATCH_SIZE//BATCH_SIZE batches"
        ),
    ] = 256
    max_steps: Annotated[int, Query(description="Max steps")] = 100000000000000000000

    d_model: Annotated[int, Query(description="Hidden size of the model")] = 512
    n_codes: Annotated[int, Query(description="Codebook size")] = 2048
    n_groups: Annotated[
        int,
        Query(
            description="Number of groups to split the latent vector into before discretization"
        ),
    ] = 16
    d_latent: Annotated[
        int, Query(description="Dimensionality of the latent space")
    ] = 1024

    lr: Annotated[
        float,
        Query(
            description="Initial learning rate, will be decayed after constant warmup of WARMUP_STEPS stepse"
        ),
    ] = 1e-4
    lr_schedule: Annotated[str, Query(description="Learning rate schedule")] = (
        "const"  # "sqrt_decay"
    )
    warmup_steps: Annotated[
        int, Query(description="Number of learning rate warmup steps")
    ] = 4000

    encoder_layers: Annotated[int, Query(description="Encoder Layers")] = 4
    decoder_layers: Annotated[int, Query(description="Decoder Layers")] = 6
    encoder_ffn_dim: Annotated[int, Query(description="Encoder FFN Dimension")] = 2048
    decoder_ffn_dim: Annotated[int, Query(description="Decoder FFN Dimension")] = 2048
    windowed_attention_pr: Annotated[
        float, Query(description="Windowed Attention PR")
    ] = 0.0
    max_lookahead: Annotated[int, Query(description="Max Lookahead")] = 4
    disable_vq: Annotated[bool, Query(description="Disable VQ")] = False

    automatic_optimization: Annotated[
        bool, Query(description="Automatic Optimization")
    ] = False
    beta: Annotated[float, Query(description="Beta")] = 0.02
    cycle_length: Annotated[int, Query(description="Cycle Length")] = 2000
    position_embedding_type: Annotated[
        str, Query(description="Position Embedding Type")
    ] = "relative_key_query"
    num_attention_heads: Annotated[
        int, Query(description="Number of Attention Heads")
    ] = 8
    decay: Annotated[float, Query(description="Decay")] = 0.995
    eps: Annotated[float, Query(description="EPS")] = 1e-4
    restart_threshold: Annotated[float, Query(description="Restart Threshold")] = 0.99


class TSNEParameters(BaseModel):
    dataset_name: Annotated[str, Query(description="Name of Dataset")] = ""
    cluster: Annotated[bool, Query(description="Cluster")] = False
    plot: Annotated[bool, Query(description="Plot")] = False

    n_components: Annotated[int, Query(description="Number of Components")] = 2
    perplexity: Annotated[float, Query(description="Perplexity")] = 30.0
    early_exaggeration: Annotated[float, Query(description="Early Exaggeration")] = 12.0
    learning_rate: Annotated[float | str, Query(description="Learning Rate")] = "auto"
    n_iter: Annotated[int, Query(description="Number of Iterations")] = 1000
    n_iter_without_progress: Annotated[
        int, Query(description="Number of Iterations Without Progress")
    ] = 300
    min_grad_norm: Annotated[float, Query(description="Minimum Gradient Norm")] = 1e-7
    metric: Annotated[str, Query(description="Metric")] = "euclidean"
    metric_params: Annotated[dict, Query(description="Metric Parameters")] = None
    init: Annotated[str, Query(description="Initialization")] = "pca"
    verbose: Annotated[int, Query(description="Verbose")] = 0
    random_state: Annotated[int, Query(description="Random State")] = None
    method: Annotated[str, Query(description="Method")] = "barnes_hut"
    angle: Annotated[float, Query(description="Angle")] = 0.5
    n_jobs: Annotated[int, Query(description="Number of Jobs")] = None


class CorrelationParameters(BaseModel):
    dataset_name: Annotated[str, Query(description="Name of Dataset")] = ""
    correlation_type: Annotated[str, Query(description="Correlation Type")] = (
        "spearmann"
    )
    rank: Annotated[bool, Query(description="Rank")] = False
    plot: Annotated[bool, Query(description="Plot")] = False
