import os
import json
import torch
import numpy as np

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback

from application.multimodal_mapping.models.input_encoders import (
    TextEncoder, 
    GlobalFeatureProcessor,
    MoodProcessor
)
from application.multimodal_mapping.models.embedding_fusion import CrossAttentionFusion, LinearConcatFusion
from application.multimodal_mapping.models.hierarchical_vae import TransformerVAE
from application.multimodal_mapping.helpers.multimodal_mapping_helper import (
    load_mapping_from_checkpoint,
    initialize_tokenizers_and_processors,
    create_multimodal_mapping_module,
)
from application.multimodal_mapping.models.multimodal_mapping import MultimodalMappingModule

from domain.constants.paths_constants import (
    CHECKPOINTS_PATH,
    GENERATED_PATH,
)
from domain.models.multimodal_mapping_model import MultimodalMappingParameters

from application.dataloader.models.dataloader_model import DataloaderModule


class KLAnnealingCallback(Callback):
    """
    Custom callback to handle KL annealing during training.
    Follows the EmoMusicTV approach of annealing the KL weight.
    """

    def __init__(self, kl_start=0.0, kl_end=1.0, kl_anneal_steps=10000):
        super().__init__()
        self.kl_start = kl_start
        self.kl_end = kl_end
        self.kl_anneal_steps = kl_anneal_steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Calculate KL weight based on current step
        if trainer.global_step >= self.kl_anneal_steps:
            kl_weight = self.kl_end
        else:
            # Linear annealing
            kl_weight = self.kl_start + (self.kl_end - self.kl_start) * (trainer.global_step / self.kl_anneal_steps)

        # Set the KL weight on the module
        pl_module.kl_weight = kl_weight


def train_multimodal_mapping(config_path: str) -> dict:
    """
    Train a multimodal mapping model using parameters from the config file.

    Args:
        config_path: Path to the configuration JSON file

    Returns:
        Dictionary with training results message
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    mapping_config = config.get("multimodal_mapping", {})

    # Create data module using the enhanced dataloader
    datamodule_parameters = config["dataloader"]
    datamodule_parameters["load_latent"] = True
    datamodule_parameters["load_symb"] = True
    datamodule_parameters["load_emotions"] = True
    datamodule_parameters["load_global_features"] = True
    datamodule_parameters["load_text_prompts"] = True

    datamodule = DataloaderModule(**datamodule_parameters)

    # Setup the datamodule
    datamodule.setup()

    # Initialize model
    if mapping_config.get("load_from_checkpoint", False):
        # Load from existing checkpoint
        weights_path = mapping_config.get("weights_path")
        training_config_path = mapping_config.get("config_path")

        # Create Lightning module
        model = MultimodalMappingModule(training_config_path)

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Create a new model
        model = MultimodalMappingModule(mapping_config)

    # Setup device
    device = torch.device(mapping_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()

    # Setup checkpointing
    training_name = mapping_config.get("training_name", "default")

    checkpoint_path = os.path.join(CHECKPOINTS_PATH, datamodule_parameters.get("dataset_name"), training_name, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=checkpoint_path,
        filename="{step}-{valid_loss:.2f}",
        save_top_k=3,
        every_n_train_steps=mapping_config.get("checkpoint_every_n_steps", 2000),
        save_last=True,
    )

    # Setup KL annealing
    kl_annealing_callback = KLAnnealingCallback(
        kl_start=mapping_config.get("kl_start", 0.0), kl_end=mapping_config.get("kl_end", 1.0), kl_anneal_steps=mapping_config.get("kl_anneal_steps", 10000)
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Setup trainer
    max_epochs = mapping_config.get("epochs", 100)
    max_steps = mapping_config.get("max_training_steps", 100000)
    val_check_interval = mapping_config.get("val_check_interval", 500)
    log_every_n_steps = mapping_config.get("log_every_n_steps", 100)

    trainer = Trainer(
        default_root_dir=os.path.join(CHECKPOINTS_PATH, datamodule_parameters.get("dataset_name"), training_name, "training_logs"),
        devices=device_count,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        profiler="simple",
        callbacks=[checkpoint_callback, lr_monitor, kl_annealing_callback],
        enable_checkpointing=True,
        max_epochs=max_epochs,
        max_steps=max_steps,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        limit_val_batches=64,
        num_sanity_val_steps=2,
    )

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_path, "final_model.ckpt")
    trainer.save_checkpoint(final_checkpoint_path)

    # Save separated checkpoint for easier loading
    output_dir = os.path.join(CHECKPOINTS_PATH, datamodule_parameters.get("dataset_name"), training_name, "separated")
    save_checkpoint_separate(final_checkpoint_path, output_dir)

    return {
        "Message": "Trained Multimodal Mapping Model",
        "final_checkpoint_path": final_checkpoint_path,
        "separated_weights_path": os.path.join(output_dir, "final_model_weights.pt"),
        "separated_config_path": os.path.join(output_dir, "final_model_config.json"),
    }


def save_checkpoint_separate(checkpoint_path: str, output_dir: str) -> dict:
    """
    Load a model checkpoint and save weights as .pt and hyperparameters as .json

    Args:
        checkpoint_path: Path to the original checkpoint file
        output_dir: Directory to save the separate files

    Returns:
        Dictionary with paths to the saved files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the original checkpoint
    pl_ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Extract hyperparameters and state dict
    hyperparameters = pl_ckpt["hyper_parameters"]
    state_dict = pl_ckpt["state_dict"]

    # Generate filenames
    base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    weights_path = os.path.join(output_dir, f"{base_name}_weights.pt")
    config_path = os.path.join(output_dir, f"{base_name}_config.json")

    # Save the files
    torch.save(state_dict, weights_path)

    with open(config_path, "w") as f:
        json.dump(hyperparameters, f, indent=2)

    return {"weights_path": weights_path, "config_path": config_path, "message": "Checkpoint successfully separated"}
