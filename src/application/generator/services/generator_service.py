import os
import pickle
import json

import torch

from src.application.encoder.helpers.remi_helper import remi2midi

from src.application.generator.models.generator_model import MIDIGeneratorModule

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.application.dataloader.models.dataloader_model import (
    DataloaderModule,
)
from src.application.generator.helper.generator_helpers import (
    load_generator_from_checkpoint,
)

from src.domain.constants.paths_constants import (
    CHECKPOINTS_PATH,
    GENERATOR_PATH,
    GENERATED_PATH,
    REPRESENTATIONS_PATH,
)
from src.domain.models.generator.generator_model import (
    GeneratorTrainingParameters,
    GeneratorGeneratePromptParameters,
)


def train_generator(parameters: GeneratorTrainingParameters) -> dict:
    torch.multiprocessing.set_start_method("spawn")

    with open("shared/assets/config.json", "r") as f:
        config = json.load(f)

    datamodule_parameters = config["dataloader"]

    datamodule_parameters["load_latent"] = parameters.load_latent
    datamodule_parameters["load_symb"] = parameters.load_symb

    print("Batch size:", flush=True)
    print(datamodule_parameters["batch_size"], flush=True)
    datamodule_parameters["batch_size"] = 4

    datamodule_parameters["context_size"] = 512
    # datamodule_parameters["max_positions"] = 512
    # datamodule_parameters["max_bars"] = 512

    # Convert train_val_test_split from list to tuple if needed
    if isinstance(datamodule_parameters["train_val_test_split"], list):
        datamodule_parameters["train_val_test_split"] = tuple(datamodule_parameters["train_val_test_split"])

    datamodule = DataloaderModule(**datamodule_parameters)

    accumulate_grad_batches = parameters.target_batch_size // datamodule.batch_size
    if parameters.load_from_checkpoint:
        generator_checkpoint = os.path.join(
            CHECKPOINTS_PATH,
            parameters.dataset_name,
            parameters.training_name,
            parameters.checkpoint_name,
        )
        model = load_generator_from_checkpoint(generator_checkpoint)
    else:
        model = MIDIGeneratorModule(
            d_model=512,  # from VAE
            d_latent=1024,  # from VAE
            n_codes=2048,  # from VAE
            n_groups=16,  # from VAE
            context_size=datamodule_parameters["context_size"],
            max_bars=datamodule_parameters["max_bars"],
            max_positions=datamodule_parameters["max_positions"],
            lr=parameters.lr,
            lr_schedule=parameters.lr_schedule,
            warmup_steps=parameters.warmup_steps,
            max_steps=parameters.max_steps,
            encoder_layers=parameters.encoder_layers,
            decoder_layers=parameters.decoder_layers,
            intermediate_size=parameters.intermediate_size,
            num_attention_heads=parameters.num_attention_heads,
            use_pretrained_latent_embeddings=parameters.use_pretrained_latent_embeddings,
            load_symb=parameters.load_symb,
            load_latent=parameters.load_latent,
            load_sentiments=parameters.load_sentiments,
            load_bert_from_ckpt=parameters.load_bert_from_ckpt,
            save_encoder_decoder_path=os.path.join(GENERATOR_PATH, parameters.dataset_name, parameters.training_name, "BERT"),
        )

    device = torch.device(parameters.device)
    model.to(device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()
    checkpoint_dir = os.path.join(CHECKPOINTS_PATH, parameters.dataset_name, parameters.training_name)
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
        default_root_dir=os.path.join(GENERATOR_PATH, parameters.dataset_name, parameters.training_name),
        devices=device_count,
        accelerator="gpu",
        profiler="simple",
        callbacks=[checkpoint_callback, lr_monitor],
        enable_checkpointing=True,
        max_epochs=parameters.epochs,
        max_steps=parameters.max_training_steps,
        log_every_n_steps=max(100, min(25 * accumulate_grad_batches, 200)),
        val_check_interval=max(500, min(300 * accumulate_grad_batches, 1000)),
        limit_val_batches=64,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)

    return {"Message": "Trained Generator"}


def generate_sample_from_prompt(parameters: GeneratorGeneratePromptParameters):
    max_bars = parameters.max_bars

    output_dir = os.path.join(GENERATED_PATH, parameters.dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generator_checkpoint = os.path.join(
        CHECKPOINTS_PATH,
        parameters.dataset_name,
        parameters.training_name,
        parameters.checkpoint_name,
    )
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
        key: tensor.unsqueeze(0)[:, : parameters.initial_context]
        for key, tensor in representation.items()
        if key not in ["file", "sentiments_vector", "input_ids", "bar_ids", "position_ids"]
    }

    # batch = None
    # sentiments_vector = parameters.label_scores
    # file, _ = find_closest_file(target_labels, file_labels)
    # print(file, flush=True)
    # sentiments_vector = torch.tensor(sentiments_vector, dtype=torch.float32, device=torch.device("cuda")).expand(1, -1)  # device = model.device
    # sentiments_vector = None

    sample = model.sample(batch, max_length=parameters.max_n_tokens, max_bars=max_bars)

    xs_hat = sample["sequences"].detach().cpu()  # Generated
    events_hat = [model.vocab.decode(x) for x in xs_hat]  # Generated

    print(events_hat)

    try:
        pm_hat = remi2midi(events_hat[0])  # Generated
    except Exception as err:
        print("ERROR: Could not convert events to midi:", err)

    if output_dir:
        pm_hat.write(os.path.join(output_dir, f"{parameters.prompt_name}.mid"))  # Generated

    return {"Message": "Generated Sample"}
