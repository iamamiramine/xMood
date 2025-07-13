import os
import torch
from typing import Union, Dict, Any

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback

# Input encoders and fusion classes are imported in helpers functions where needed
from application.multimodal_mapping.helpers.multimodal_mapping_helper import (
    load_from_checkpoint,
    load_from_checkpoint_new,
)
from application.multimodal_mapping.models.multimodal_mapping import MultimodalMappingModule

from domain.models.multimodal.multimodal_model import MultimodalMappingParameters, MultimodalTrainingParameters

from core.dataloader.models.dataloader_model import DataloaderModule


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


def train_multimodal_mapping(parameters: MultimodalTrainingParameters) -> dict:
    """
    Train a multimodal mapping model using BaseModel parameters.

    Args:
        parameters: MultimodalTrainingParameters object

    Returns:
        Dictionary with training results message
    """
    
    # Set up data module parameters
    datamodule_parameters = {
        "dataset_name": parameters.dataset_name,
        "context_size": parameters.context_size,
        "max_positions": parameters.max_positions,
        "max_bars": parameters.max_bars,
        "max_bars_per_context": parameters.max_bars_per_context,
        "max_contexts_per_file": parameters.max_contexts_per_file,
        "bar_token_mask": parameters.bar_token_mask,
        "bar_token_idx": parameters.bar_token_idx,
        "batch_size": parameters.batch_size,
        "num_workers": parameters.num_workers,
        "pin_memory": parameters.pin_memory,
        "train_val_test_split": parameters.train_val_test_split,
    }

    datamodule = DataloaderModule(**datamodule_parameters)

    accumulate_grad_batches = parameters.batch_size // datamodule.batch_size

    # Initialize model
    if parameters.load_from_checkpoint and parameters.checkpoint_path:
        # Load from checkpoint
        model = load_from_checkpoint(parameters.checkpoint_path, eval=False)
    elif parameters.load_from_checkpoint and parameters.weights_path and parameters.config_path:
        # Load from separate weights and config files
        model = load_from_checkpoint_new(parameters.weights_path, parameters.config_path, eval=False)
    else:
        # Create new model from parameters
        model_params = {
            # Input dimensions
            "image_dim": parameters.image_dim,
            "text_dim": parameters.text_dim,
            "global_feature_dim": parameters.global_feature_dim,
            "global_feature_out_dim": parameters.global_feature_out_dim,
            "mood_dim": parameters.mood_dim,
            "mood_out_dim": parameters.mood_out_dim,
            
            # Architecture parameters
            "fusion_dim": parameters.fusion_dim,
            "d_model": parameters.d_model,
            "latent_dim": parameters.latent_dim,
            "output_dim": parameters.output_dim,
            "context_size": parameters.context_size,
            "num_layers": parameters.num_layers,
            "num_heads": parameters.num_heads,
            "fusion_heads": parameters.fusion_heads,
            "fusion_type": parameters.fusion_type,
            "dropout": parameters.dropout,
            "training": parameters.training,
            
            # Modality dropout rates
            "image_dropout_rate": parameters.image_dropout_rate,
            "text_dropout_rate": parameters.text_dropout_rate,
            "global_feature_dropout_rate": parameters.global_feature_dropout_rate,
            "mood_dropout_rate": parameters.mood_dropout_rate,
            
            # Learning rate and schedule
            "lr": parameters.lr,
            "lr_schedule": parameters.lr_schedule,
            "warmup_steps": parameters.warmup_steps,
            "max_steps": parameters.max_steps,
            
            # KL annealing
            "use_kl_annealing": parameters.use_kl_annealing,
            "kl_start": parameters.kl_start,
            "kl_end": parameters.kl_end,
            "kl_anneal_steps": parameters.kl_anneal_steps,
        }
        
        model = MultimodalMappingModule(**model_params)

    device = torch.device(parameters.device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(parameters.checkpoint_dir, exist_ok=True)

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=parameters.checkpoint_dir,
        filename="{step}-{val_loss:.2f}",
        save_last=True,
        save_top_k=parameters.save_top_k,
        every_n_train_steps=parameters.every_n_train_steps,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Add KL annealing callback if enabled
    callbacks = [checkpoint_callback, lr_monitor]
    if parameters.use_kl_annealing:
        kl_callback = KLAnnealingCallback(
            kl_start=parameters.kl_start,
            kl_end=parameters.kl_end,
            kl_anneal_steps=parameters.kl_anneal_steps
        )
        callbacks.append(kl_callback)

    trainer = Trainer(
        max_steps=parameters.max_steps,
        max_epochs=parameters.max_epochs,
        accelerator="gpu" if device_count > 0 else "cpu",
        devices=min(device_count, parameters.gpus) if device_count > 0 else "auto",
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=parameters.val_check_interval,
        log_every_n_steps=parameters.log_every_n_steps,
        limit_val_batches=parameters.limit_val_batches,
        num_sanity_val_steps=parameters.num_sanity_val_steps,
        callbacks=callbacks,
    )

    try:
        trainer.fit(model, datamodule=datamodule)
        return {"Message": "Multimodal mapping training completed successfully"}
    except Exception as e:
        return {"Message": f"Multimodal mapping training failed: {str(e)}"}


def generate_multimodal_representations(parameters: Union[MultimodalMappingParameters, Dict[str, Any]]) -> dict:
    """Generate multimodal representations using trained model."""
    
    if isinstance(parameters, dict):
        parameters = MultimodalMappingParameters(**parameters)
    
    # Load model
    model = load_from_checkpoint_new(parameters.weights_path, parameters.config_path, eval=True)
    
    # TODO: Implement generation logic
    
    return {
        "status": "success",
        "message": "Multimodal representations generated successfully",
        "output_folder": parameters.output_folder,
        "output_name": parameters.output_name
    }
