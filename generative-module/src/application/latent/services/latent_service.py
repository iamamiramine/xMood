import os
import json

import torch

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from core.dataloader.models.dataloader_model import DataloaderModule
from application.latent.helpers.latent_features_helper import (
    load_vae_from_checkpoint,
)
from application.latent.models.vae_model import VqVaeModule
from domain.models.latent.latent_model import (
    VaeTrainingParameters,
    LatentRepresentationParameters,
)


def train_vae(parameters: VaeTrainingParameters) -> dict:
    """Train a VAE model using BaseModel parameters."""

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

    if parameters.load_from_checkpoint and parameters.checkpoint_path:
        model = load_vae_from_checkpoint(parameters.checkpoint_path)
    else:
        # Create new model from parameters
        model_params = {
            "dataset_name": parameters.dataset_name,
            "d_model": parameters.d_model,
            "context_size": parameters.context_size,
            "n_codes": parameters.n_codes,
            "n_groups": parameters.n_groups,
            "d_latent": parameters.d_latent,
            "lr": parameters.lr,
            "lr_schedule": parameters.lr_schedule,
            "warmup_steps": parameters.warmup_steps,
            "max_steps": parameters.max_steps,
            "encoder_layers": parameters.encoder_layers,
            "decoder_layers": parameters.decoder_layers,
            "encoder_ffn_dim": parameters.encoder_ffn_dim,
            "decoder_ffn_dim": parameters.decoder_ffn_dim,
            "windowed_attention_pr": parameters.windowed_attention_pr,
            "max_lookahead": parameters.max_lookahead,
            "disable_vq": parameters.disable_vq,
            "accumulate_grad_batches": accumulate_grad_batches,
            "max_positions": parameters.max_positions,
            "automatic_optimization": parameters.automatic_optimization,
            "beta": parameters.beta,
            "cycle_length": parameters.cycle_length,
            "position_embedding_type": parameters.position_embedding_type,
            "num_attention_heads": parameters.num_attention_heads,
            "decay": parameters.decay,
            "eps": parameters.eps,
            "restart_threshold": parameters.restart_threshold,
        }

        model = VqVaeModule(**model_params)

    device = torch.device(parameters.device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(parameters.checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=parameters.checkpoint_dir,
        filename="{step}-{val_loss:.2f}",
        save_last=True,
        save_top_k=parameters.save_top_k,
        every_n_train_steps=parameters.every_n_train_steps,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = WandbLogger(project="vae-training") if hasattr(parameters, "use_wandb") and parameters.use_wandb else None

    trainer = L.Trainer(
        max_steps=parameters.max_steps,
        max_epochs=parameters.max_epochs,
        accelerator="gpu" if device_count > 0 else "cpu",
        devices=min(device_count, parameters.gpus) if device_count > 0 else "auto",
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=parameters.val_check_interval,
        log_every_n_steps=parameters.log_every_n_steps,
        limit_val_batches=parameters.limit_val_batches,
        num_sanity_val_steps=parameters.num_sanity_val_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
    )

    try:
        trainer.fit(model, datamodule=datamodule)
        return {"Message": "VAE training completed successfully"}
    except Exception as e:
        return {"Message": f"VAE training failed: {str(e)}"}


async def generate_latent_representations_dataset(parameters: LatentRepresentationParameters) -> dict:
    """Generate latent representations using BaseModel parameters."""

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
    checkpoint_path = parameters.checkpoint_path

    model = load_vae_from_checkpoint(checkpoint_path)

    device = torch.device(parameters.device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()

    if not os.path.exists(parameters.output_dir):
        os.makedirs(parameters.output_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=parameters.output_dir,
        filename="{step}-{valid_loss:.2f}",
        save_last=True,
        save_top_k=parameters.save_top_k,
        every_n_train_steps=parameters.every_n_train_steps,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        max_steps=parameters.max_steps,
        accelerator="gpu" if device_count > 0 else "cpu",
        devices=min(device_count, 1) if device_count > 0 else "auto",
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=parameters.val_check_interval,
        log_every_n_steps=parameters.log_every_n_steps,
        limit_val_batches=parameters.limit_val_batches,
        num_sanity_val_steps=parameters.num_sanity_val_steps,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    try:
        predictions = trainer.predict(model, datamodule=datamodule)

        # Process and save predictions if requested
        if parameters.save_latents or parameters.save_codes:
            processed_predictions = []
            for batch_predictions in predictions:
                for prediction in batch_predictions:
                    if isinstance(prediction, dict):
                        processed_predictions.append(prediction)

            # Save predictions to the output directory
            output_file = os.path.join(parameters.output_dir, "latent_representations.json")
            with open(output_file, "w") as f:
                json.dump(processed_predictions, f, indent=2)

        return {"Message": f"Latent representation generation completed. {len(predictions)} batches processed."}
    except Exception as e:
        return {"Message": f"Latent representation generation failed: {str(e)}"}
