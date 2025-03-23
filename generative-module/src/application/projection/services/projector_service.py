import os
import pickle
import json

import torch

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from application.dataloader.models.dataloader_model import DataloaderModule
from application.encoder.models.vocab_model import MoodsVocab, SymbolicFeaturesVocab
from application.projection.helpers.projector_helper import load_projector_from_checkpoint
from application.projection.models.projection_model import ProjectorModule

from domain.constants.paths_constants import (
    CHECKPOINTS_PATH,
    GENERATED_PATH,
    REPRESENTATIONS_PATH,
    PROCESSED_PATH,
)
from persistence.dataloader.repositories.dataloader_repository import async_load


def train_projector(config_path: str) -> dict:
    """Train a projector model using parameters from the config file."""
    torch.multiprocessing.set_start_method("spawn")

    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    projector_config = config.get("projector", {})
    datamodule_parameters = config["dataloader"]

    # Convert train_val_test_split from list to tuple if needed
    if isinstance(datamodule_parameters["train_val_test_split"], list):
        datamodule_parameters["train_val_test_split"] = tuple(datamodule_parameters["train_val_test_split"])

    datamodule = DataloaderModule(**datamodule_parameters)

    accumulate_grad_batches = projector_config.get("target_batch_size", 256) // datamodule.batch_size
    if projector_config.get("load_from_checkpoint", False):
        projector_checkpoint = projector_config["checkpoint_path"]
        model = load_projector_from_checkpoint(projector_checkpoint)
    else:
        model = ProjectorModule(
            d_model=config.get("vae", {}).get("d_model", 512),
            d_latent=config.get("vae", {}).get("d_latent", 1024),
            context_size=datamodule_parameters["context_size"],
            max_bars=datamodule_parameters["max_bars"],
            max_positions=datamodule_parameters["max_positions"],
            lr=projector_config.get("lr", 1e-4),
            lr_schedule=projector_config.get("lr_schedule", "const"),
            warmup_steps=projector_config.get("warmup_steps", 4000),
            max_steps=projector_config.get("max_steps", 100000000000000000000),
            encoder_layers=projector_config.get("encoder_layers", 6),
            decoder_layers=projector_config.get("decoder_layers", 6),
            intermediate_size=projector_config.get("intermediate_size", 2048),
            num_attention_heads=projector_config.get("num_attention_heads", 8),
            device=projector_config.get("device", "cuda:0"),
        )

    device = torch.device(projector_config.get("device", "cuda"))
    model.to(device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()
    checkpoint_path = os.path.join(CHECKPOINTS_PATH, datamodule_parameters.get("dataset_name"), projector_config.get("training_name"), "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=checkpoint_path,
        filename="{step}-{valid_loss:.2f}",
        save_last=True,
        save_top_k=0,
        every_n_train_steps=100,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        default_root_dir=os.path.join(CHECKPOINTS_PATH, datamodule_parameters.get("dataset_name"), projector_config.get("training_name"), "training_logs"),
        devices=device_count,
        accelerator="gpu",
        profiler="simple",
        callbacks=[checkpoint_callback, lr_monitor],
        enable_checkpointing=True,
        max_epochs=projector_config.get("epochs", 100),
        max_steps=projector_config.get("max_training_steps", 100000),
        log_every_n_steps=max(100, min(25 * accumulate_grad_batches, 200)),
        val_check_interval=max(500, min(300 * accumulate_grad_batches, 1000)),
        limit_val_batches=64,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)

    return {"Message": "Trained Projector"}


def generate_features_from_prompt(config_path: str):
    """Generate features using parameters from the config file."""
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    projector_config = config.get("projector", {})
    prompt_config = projector_config.get("generate", {})

    output_dir = os.path.join(GENERATED_PATH, prompt_config.get("output_folder"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    projector_checkpoint = projector_config.get("checkpoint_path")
    model = load_projector_from_checkpoint(projector_checkpoint)
    model = model.to("cuda")

    processed_dir = os.path.join(PROCESSED_PATH, "Emotion4MIDI_Sample")

    # Load the representation file
    file = prompt_config.get("input_file", "input_representation")
    processed_data = async_load(processed_dir, file, "processed")

    emotion_vocab = MoodsVocab()
    symb_vocab = SymbolicFeaturesVocab()

    piece_emotions_tokens = processed_data["emotions"]["piece_emotions"]["piece_emotions_tokens"]
    piece_emotions_tokens = torch.tensor(emotion_vocab.encode(piece_emotions_tokens), dtype=torch.int)

    piece_symbolic = processed_data["symbolic_features"]["piece_symbolic"]
    piece_symbolic = torch.tensor(symb_vocab.encode(piece_symbolic), dtype=torch.int)

    # Ensure consistent sequence lengths by padding to the maximum length
    max_seq_len = max(piece_emotions_tokens.size(0), piece_symbolic.size(0))

    # Pad piece_emotions_tokens if needed
    if piece_emotions_tokens.size(0) < max_seq_len:
        padding = torch.zeros(max_seq_len - piece_emotions_tokens.size(0), dtype=torch.int, device=piece_emotions_tokens.device)
        piece_emotions_tokens = torch.cat([piece_emotions_tokens, padding])

    # Pad piece_symbolic if needed
    if piece_symbolic.size(0) < max_seq_len:
        padding = torch.zeros(max_seq_len - piece_symbolic.size(0), dtype=torch.int, device=piece_symbolic.device)
        piece_symbolic = torch.cat([piece_symbolic, padding])

    # Add batch dimension
    piece_emotions_tokens = piece_emotions_tokens.unsqueeze(0)
    piece_symbolic = piece_symbolic.unsqueeze(0)

    # Prepare the batch with the conditioning information
    batch = {"piece_emotions_ids": piece_emotions_tokens, "piece_symbolic_ids": piece_symbolic}

    # Generate features
    generated_features = model.sample(batch)

    # Decode the generated sequences
    sequences = generated_features["sequences"].cpu().numpy()
    decoded_sequences = []

    for sequence in sequences:
        decoded_tokens = symb_vocab.decode(sequence.tolist())
        # Post-process the decoded sequence
        decoded_tokens = model.post_process_sequence(decoded_tokens)
        decoded_sequences.append(decoded_tokens)

    print("Original piece symbolic features:", flush=True)
    print(processed_data["symbolic_features"]["piece_symbolic"], flush=True)
    print("\nOriginal bar symbolic features:", flush=True)
    print(processed_data["symbolic_features"]["bar_symbolic"], flush=True)
    print("\nGenerated and post-processed sequences:", flush=True)
    print(decoded_sequences, flush=True)

    # # Save both raw and decoded features
    # output_file = os.path.join(output_dir, f"{prompt_config.get('prompt_name', 'generated_features')}.pkl")
    # with open(output_file, "wb") as f:
    #     pickle.dump({
    #         'raw_sequences': generated_features['sequences'].cpu(),
    #         'decoded_sequences': decoded_sequences
    #     }, f)

    return {"Message": "Generated Features"}
