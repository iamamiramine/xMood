import os
import pickle
import json
import glob
import tempfile
import shutil

import torch
from torch.nn.utils.rnn import pad_sequence

from application.dataloader.helper.dataloader_helper import represent_encoding
from application.encoder.helpers.remi_helper import remi2midi
from application.encoder.helpers.vocab_helper import get_positions, get_bars, mask_bar_tokens, get_bos_eos_events
from application.encoder.models.vocab_model import RemiVocab, SymbolicFeaturesVocab, MoodsVocab

from application.generator.models.generator_model import MIDIGeneratorModule
from domain.models.generator_model import GenerateFromMIDIParameters

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback

from application.dataloader.models.dataloader_model import (
    DataloaderModule,
)
from application.generator.helper.generator_helpers import (
    load_generator_from_checkpoint,
    load_from_checkpoint_new,
)

from domain.constants.paths_constants import (
    CHECKPOINTS_PATH,
    GENERATED_PATH,
    PROCESSED_PATH,
    LATENTS_PATH,
)
from domain.constants.encoder.token_constants import (
    PAD_TOKEN,
    EOS_TOKEN,
    BAR_KEY,
    BOS_TOKEN,
)
from persistence.dataloader.repositories.dataloader_repository import async_load, CPU_Unpickler


class CheckpointCleanupCallback(Callback):
    """Custom callback to clean up temporary checkpoint files."""

    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        # Get system temp directory
        self.temp_dir = tempfile.gettempdir()

    def cleanup_temp_files(self):
        # Clean checkpoint dir
        temp_files = glob.glob(os.path.join(self.checkpoint_path, "tmp_*"))
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except (OSError, FileNotFoundError):
                pass

        # Clean system temp directory
        for root, dirs, files in os.walk(self.temp_dir):
            for file in files:
                if file.startswith("tmp") and file.endswith(".ckpt"):
                    try:
                        full_path = os.path.join(root, file)
                        os.remove(full_path)
                    except (OSError, FileNotFoundError):
                        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Clean up temporary files every 1000 steps
        if trainer.global_step % 1000 == 0:
            self.cleanup_temp_files()

    def on_train_end(self, trainer, pl_module):
        # Final cleanup at the end of training
        self.cleanup_temp_files()


def train_generator(config_path: str) -> dict:
    """Train a generator model using parameters from the config file."""
    torch.multiprocessing.set_start_method("spawn")

    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    generator_config = config.get("generator", {})
    datamodule_parameters = config["dataloader"]
    datamodule_parameters["encode"] = False

    # Convert train_val_test_split from list to tuple if needed
    if isinstance(datamodule_parameters["train_val_test_split"], list):
        datamodule_parameters["train_val_test_split"] = tuple(datamodule_parameters["train_val_test_split"])

    datamodule = DataloaderModule(**datamodule_parameters)

    accumulate_grad_batches = generator_config.get("target_batch_size", 256) // datamodule.batch_size
    generator_checkpoint = generator_config.get("checkpoint_path")

    if generator_config.get("load_from_checkpoint", False):
        weights_path = generator_config.get("weights_path")
        training_config_path = generator_config.get("config_path")
        model = load_from_checkpoint_new(weights_path, training_config_path, eval=False)

        # # Original loading method
        # model = load_generator_from_checkpoint(generator_checkpoint, eval=False)
    else:
        model = MIDIGeneratorModule(
            d_model=config.get("vae", {}).get("d_model", 512),
            d_latent=config.get("vae", {}).get("d_latent", 1024),
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
        )

    device = torch.device(generator_config.get("device", "cuda"))
    model.to(device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()
    checkpoint_path = os.path.join(CHECKPOINTS_PATH, datamodule_parameters.get("dataset_name"), generator_config.get("training_name"), "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=checkpoint_path,
        filename="{step}-{valid_loss:.2f}",
        save_last=True,
        save_top_k=0,
        every_n_train_steps=5000,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    cleanup_callback = CheckpointCleanupCallback(checkpoint_path)

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


def generate_from_midi(parameters: GenerateFromMIDIParameters) -> dict:
    """Generate a sample using existing MIDI files as prompts."""

    # Load the model from checkpoint
    weights_path = parameters.weights_path
    training_config_path = parameters.config_path
    model = load_from_checkpoint_new(weights_path, training_config_path)
    
    # # Original loading method
    # model = load_generator_from_checkpoint(parameters.checkpoint_path)
    
    model = model.to("cuda")

    processed_dir = os.path.join(PROCESSED_PATH, "MIDICaps")

    # Load latents
    latent_data = async_load(processed_dir, parameters.latent_midi, "processed")
    latents_path = os.path.join(
        str(LATENTS_PATH),
        "MIDICaps",
        f"{os.path.basename(parameters.latent_midi)}_latents.pkl",
    )
    latents_file = CPU_Unpickler(open(latents_path, "rb")).load()

    # Load symbolic and emotions data
    symbolic_data = async_load(processed_dir, parameters.symbolic_midi, "processed")
    # emotions_data = async_load(processed_dir, parameters.emotions_midi, "processed")

    # Prepare batch for inference
    latent_rep = represent_encoding(
        parameters.latent_midi,
        "",
        parameters.context_size,
        parameters.max_bars,
        parameters.max_positions,
        None,
        -1,
        -1,
        events=latent_data["encodings"]["events"],
        latents=latents_file["latents"],
        codes=latents_file["codes"],
        bar_symbolic=None,
        piece_symbolic=None,
        save=False,
    )
    latents = latent_rep["latents"]

    symb_rep = represent_encoding(
        parameters.symbolic_midi,
        "",
        parameters.context_size,
        parameters.max_bars,
        parameters.max_positions,
        None,
        -1,
        -1,
        events=symbolic_data["encodings"]["events"],
        latents=None,
        codes=None,
        bar_symbolic=symbolic_data["symbolic_features"]["bar_symbolic"],
        piece_symbolic=None,
        save=False,
    )
    symb_bar_ids = symb_rep["symb_bar_ids"]
    symb_ids = symb_rep["bar_symbolic"]

    # emotions_rep = represent_encoding(
    #     parameters.emotions_midi,
    #     "",
    #     parameters.context_size,
    #     parameters.max_bars,
    #     parameters.max_positions,
    #     None,
    #     -1,
    #     -1,
    #     events=emotions_data["encodings"]["events"],
    #     latents=None,
    #     codes=None,
    #     bar_symbolic=None,
    #     piece_symbolic=None,
    #     piece_emotions_vector=None,
    #     piece_emotions_tokens=emotions_data["emotions"]["piece_emotions"]["piece_emotions_tokens"],
    #     save=False,
    # )
    # emotions_ids = emotions_rep["piece_emotions_ids"]

    representation = {
        "latents": latents,
        "bar_symbolic": symb_ids,
        "symb_bar_ids": symb_bar_ids,
        # "piece_emotions_ids": emotions_ids,
    }

    batch = {key: tensor.unsqueeze(0)[:, :256] for key, tensor in representation.items()}  # parameters.initial_context

    # Generate the sample
    sample = model.sample(batch, max_length=parameters.max_n_tokens, max_bars=parameters.max_bars, temp=parameters.temperature)

    xs_hat = sample["sequences"].detach().cpu()
    events_hat = [model.vocab.decode(x) for x in xs_hat]

    try:
        pm_hat = remi2midi(events_hat[0])
    except Exception as err:
        return {"Message": f"ERROR: Could not convert events to midi: {err}"}

    # Create output directory if it doesn't exist
    output_dir = os.path.join(GENERATED_PATH, parameters.output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the generated MIDI
    output_path = os.path.join(output_dir, f"{parameters.output_name}.mid")
    pm_hat.write(output_path)

    return {"Message": "Generated Sample", "output_path": output_path}


def save_checkpoint_separate(checkpoint_path: str, output_dir: str) -> dict:
    """
    Load a model checkpoint (.pkl or .ckpt) and save weights as .pt and hyperparameters as .json
    
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
    
    # Filter out position_ids (same as in the original loader)
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith("embeddings.position_ids")}
    
    # Generate filenames (using the original filename as a base)
    base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    weights_path = os.path.join(output_dir, f"{base_name}_weights.pt")
    config_path = os.path.join(output_dir, f"{base_name}_config.json")
    
    # Save the files
    torch.save(state_dict, weights_path)
    
    with open(config_path, "w") as f:
        json.dump(hyperparameters, f, indent=2)
        
    return {
        "weights_path": weights_path,
        "config_path": config_path,
        "message": "Checkpoint successfully separated"
    }
