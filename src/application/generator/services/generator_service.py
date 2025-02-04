import os
import pickle
import json

import torch

from application.encoder.helpers.remi_helper import remi2midi

from application.generator.models.generator_model import MIDIGeneratorModule

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from application.dataloader.models.dataloader_model import (
    DataloaderModule,
)
from application.generator.helper.generator_helpers import (
    load_generator_from_checkpoint,
)

from domain.constants.paths_constants import (
    CHECKPOINTS_PATH,
    GENERATED_PATH,
    REPRESENTATIONS_PATH,
)


def train_generator(config_path: str) -> dict:
    """Train a generator model using parameters from the config file."""
    torch.multiprocessing.set_start_method("spawn")

    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    generator_config = config.get("generator", {})
    datamodule_parameters = config["dataloader"]

    datamodule_parameters["batch_size"] = 4
    datamodule_parameters["context_size"] = 512

    # Convert train_val_test_split from list to tuple if needed
    if isinstance(datamodule_parameters["train_val_test_split"], list):
        datamodule_parameters["train_val_test_split"] = tuple(datamodule_parameters["train_val_test_split"])

    datamodule = DataloaderModule(**datamodule_parameters)

    accumulate_grad_batches = generator_config.get("target_batch_size", 256) // datamodule.batch_size
    if generator_config.get("load_from_checkpoint", False):
        generator_checkpoint = os.path.join(
            CHECKPOINTS_PATH,
            datamodule_parameters.get("dataset_name"),
            generator_config.get("training_name"),
            generator_config.get("checkpoint_name"),
        )
        model = load_generator_from_checkpoint(generator_checkpoint)
    else:
        model = MIDIGeneratorModule(
            d_model=config.get("vae", {}).get("d_model", 512),
            d_latent=config.get("vae", {}).get("d_latent", 1024),
            n_codes=config.get("vae", {}).get("n_codes", 2048),
            n_groups=config.get("vae", {}).get("n_groups", 16),
            context_size=datamodule_parameters["context_size"],
            max_bars=datamodule_parameters["max_bars"],
            max_positions=datamodule_parameters["max_positions"],
            lr=generator_config.get("lr", 1e-4),
            lr_schedule=generator_config.get("lr_schedule", "const"),
            warmup_steps=generator_config.get("warmup_steps", 4000),
            max_steps=generator_config.get("max_steps", 100000000000000000000),
            encoder_layers=generator_config.get("encoder_layers", 6),
            decoder_layers=generator_config.get("decoder_layers", 6),
            intermediate_size=generator_config.get("intermediate_size", 2048),
            num_attention_heads=generator_config.get("num_attention_heads", 8),
            use_pretrained_latent_embeddings=generator_config.get("use_pretrained_latent_embeddings", True),
        )

    device = torch.device(generator_config.get("device", "cuda"))
    model.to(device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()
    checkpoint_dir = os.path.join(CHECKPOINTS_PATH, datamodule_parameters.get("dataset_name"), generator_config.get("training_name"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=checkpoint_dir,
        filename="{step}-{valid_loss:.2f}",
        save_last=True,
        save_top_k=2,
        every_n_train_steps=100,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        default_root_dir=os.path.join(CHECKPOINTS_PATH, datamodule_parameters.get("dataset_name"), generator_config.get("training_name"), "training_logs"),
        devices=device_count,
        accelerator="gpu",
        profiler="simple",
        callbacks=[checkpoint_callback, lr_monitor],
        enable_checkpointing=True,
        max_epochs=generator_config.get("epochs", 100),
        max_steps=generator_config.get("max_training_steps", 100000),
        log_every_n_steps=max(100, min(25 * accumulate_grad_batches, 200)),
        val_check_interval=max(500, min(300 * accumulate_grad_batches, 1000)),
        limit_val_batches=64,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)

    return {"Message": "Trained Generator"}


def generate_sample_from_prompt(config_path: str):
    """Generate a sample using parameters from the config file."""
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    generator_config = config.get("generator", {})
    prompt_config = generator_config.get("generate", {})

    output_dir = os.path.join(GENERATED_PATH, prompt_config.get("output_folder"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generator_checkpoint = generator_config.get("checkpoint_path")
    model = load_generator_from_checkpoint(generator_checkpoint)
    model = model.to("cuda")

    # target_labels = {}
    # for i, key in enumerate(parameters.label_names):
    #     target_labels[key] = parameters.label_scores[i]
    #
    # labels, label_columns = read_labels(parameters.dataset_name, split="train")
    #
    # file_labels = {}
    # for label in target_labels:
    #     for midi in labels["file_name"].values:
    #         row = labels[labels["file_name"] == midi]
    #         if midi not in file_labels:
    #             file_labels[midi] = {}
    #         file_labels[midi][label] = row[label].values[0]
    #
    # file, _ = find_closest_file(target_labels, file_labels)

    file = "0a0a52a35fa8a6384ef810e82d5ba53c.mid__representation"
    # file = "05db7160389e20ad1ba447a9798d0025.mid__representation"

    representation = pickle.load(
        open(
            os.path.join(
                REPRESENTATIONS_PATH,
                parameters.dataset_name,
                f"{os.path.basename(file)}.pkl",
            ),
            "rb",
        )
    )

    batch = {
        key: tensor.unsqueeze(0)[:, : prompt_config.get("context_size", 256)]
        for key, tensor in representation.items()
        if key not in ["file", "sentiments_vector", "input_ids", "bar_ids", "position_ids"]
    }

    # batch = None
    # sentiments_vector = parameters.label_scores
    # file, _ = find_closest_file(target_labels, file_labels)
    # print(file, flush=True)
    # sentiments_vector = torch.tensor(sentiments_vector, dtype=torch.float32, device=torch.device("cuda")).expand(1, -1)  # device = model.device
    # sentiments_vector = None

    sample = model.sample(batch, max_length=prompt_config.get("max_n_tokens", 1024), max_bars=prompt_config.get("max_bars", 16))

    xs_hat = sample["sequences"].detach().cpu()  # Generated
    events_hat = [model.vocab.decode(x) for x in xs_hat]  # Generated

    print(events_hat)

    try:
        pm_hat = remi2midi(events_hat[0])  # Generated
    except Exception as err:
        print("ERROR: Could not convert events to midi:", err)

    if output_dir:
        pm_hat.write(os.path.join(output_dir, f"{prompt_config.get('prompt_name', 'generated_sample')}.mid"))  # Generated

    return {"Message": "Generated Sample"}
