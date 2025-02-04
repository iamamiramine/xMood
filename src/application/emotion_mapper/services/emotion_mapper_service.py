# Standard library imports
import os
import random
import json
import pickle
from itertools import chain

# PyTorch and related imports
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from persistence.dataloader.repositories.dataloader_repository import async_load, save_async

# Vocabulary models
from application.encoder.models.vocab_model import RemiVocab, EmotionVocab

# Data loading and processing
from application.dataloader.models.dataloader_model import (
    DataloaderModule,
    DataloaderDataset,
)
from application.dataloader.models.dataloader_seq_collator_model import SeqCollator

# Emotion mapper components
from application.emotion_mapper.helpers.emotion_mapper_helper import (
    load_emotion_mapper_from_checkpoint,
    convert_emotions_to_sequence,
    read_labels,
)
from application.emotion_mapper.models.emotion_mapper_model import EmotionMapper

# Constants and configuration
from domain.constants.paths_constants import (
    CHECKPOINTS_PATH,
    EMOTION_MAPPING_PATH,
    PROCESSED_PATH,
)

# Domain models
from domain.models.emotion_mapper.emotion_mapper_model import (
    EmotionMapperTrainingParameters,
    EmotionMapperGenerateParameters,
)

# Emotion keys
from domain.constants.encoder.token_constants import (
    ANGER_KEY,
    LOVE_KEY,
    SADNESS_KEY,
    JOY_KEY,
    SURPRISE_KEY,
    ADMIRATION_KEY,
    AMUSEMENT_KEY,
    ANNOYANCE_KEY,
    APPROVAL_KEY,
    CARING_KEY,
    CONFUSION_KEY,
    CURIOSITY_KEY,
    DESIRE_KEY,
    DISAPPOINTMENT_KEY,
    DISAPPROVAL_KEY,
    DISGUST_KEY,
    EMBARRASSMENT_KEY,
    EXCITEMENT_KEY,
    FEAR_KEY,
    GRATITUDE_KEY,
    GRIEF_KEY,
    NERVOUSNESS_KEY,
    NEUTRAL_KEY,
    OPTIMISM_KEY,
    PRIDE_KEY,
    REALIZATION_KEY,
    RELIEF_KEY,
    REMORSE_KEY,
)


def extract_emotion_vectors(dataset_name: str) -> dict:
    """Extract emotion vectors for all MIDI files in a dataset and save them to processed files.

    Args:
        dataset_name: Name of the dataset to process

    Returns:
        Dictionary containing the processing summary
    """
    processed_dir = os.path.join(PROCESSED_PATH, dataset_name)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Read emotion labels
    labels_df, label_columns = read_labels(dataset_name)

    # Get all emotion keys from token_constants that are present in label_columns
    emotion_key_mapping = {
        "anger": ANGER_KEY,
        "love": LOVE_KEY,
        "sadness": SADNESS_KEY,
        "joy": JOY_KEY,
        "surprise": SURPRISE_KEY,
        "admiration": ADMIRATION_KEY,
        "amusement": AMUSEMENT_KEY,
        "annoyance": ANNOYANCE_KEY,
        "approval": APPROVAL_KEY,
        "caring": CARING_KEY,
        "confusion": CONFUSION_KEY,
        "curiosity": CURIOSITY_KEY,
        "desire": DESIRE_KEY,
        "disappointment": DISAPPOINTMENT_KEY,
        "disapproval": DISAPPROVAL_KEY,
        "disgust": DISGUST_KEY,
        "embarrassment": EMBARRASSMENT_KEY,
        "excitement": EXCITEMENT_KEY,
        "fear": FEAR_KEY,
        "gratitude": GRATITUDE_KEY,
        "grief": GRIEF_KEY,
        "nervousness": NERVOUSNESS_KEY,
        "neutral": NEUTRAL_KEY,
        "optimism": OPTIMISM_KEY,
        "pride": PRIDE_KEY,
        "realization": REALIZATION_KEY,
        "relief": RELIEF_KEY,
        "remorse": REMORSE_KEY,
    }

    processed = 0
    errors = []

    # Process each file in the dataset
    for idx, row in labels_df.iterrows():
        try:
            file_name = row["file_name"]

            # Create emotion vector with values for each emotion present in labels
            emotion_vector = []
            emotion_tokens = []

            # Process each emotion label that exists in our data
            for label in label_columns:
                if label.lower() in emotion_key_mapping:
                    value = float(row[label])  # Get the emotion value
                    emotion_key = emotion_key_mapping[label.lower()]
                    emotion_tokens.append(f"{emotion_key}_{value:.4f}")
                    emotion_vector.append(value)

            # Load existing processed data
            try:
                processed_data = async_load(processed_dir, file_name, "processed")
                processed_data["emotions"] = {}
            except FileNotFoundError:
                # Skip if processed file doesn't exist yet
                continue

            # Add emotion vector and tokens to processed data
            processed_data["emotions"]["piece_emotions"] = {
                "piece_emotions_vector": emotion_vector,
                "piece_emotions_tokens": emotion_tokens,
            }

            # Save back to processed file
            save_async(processed_dir, file_name, processed_data, "processed")
            processed += 1

        except Exception as e:
            errors.append(f"Error processing {file_name}: {str(e)}")

    return {"Message": "Emotion Vector Extraction Complete"}


def train_emotion_mapper(parameters: EmotionMapperTrainingParameters) -> dict:
    """Train the emotion mapper model"""
    torch.multiprocessing.set_start_method("spawn")

    # Load datamodule parameters
    with open("shared/assets/config.json", "r") as f:
        config = json.load(f)

    datamodule_parameters = config["dataloader"]

    # Update datamodule parameters
    datamodule_parameters["batch_size"] = parameters.batch_size
    datamodule_parameters["context_size"] = 512

    # Initialize datamodule
    datamodule = DataloaderModule(**datamodule_parameters)

    # Initialize or load model
    if parameters.load_from_checkpoint:
        emotion_mapper_checkpoint = os.path.join(
            CHECKPOINTS_PATH,
            parameters.dataset_name,
            parameters.training_name,
            parameters.checkpoint_name,
        )
        model = load_emotion_mapper_from_checkpoint(emotion_mapper_checkpoint)
    else:
        model = EmotionMapper(d_model=parameters.d_model, num_heads=parameters.num_heads)

    # Setup device
    device = torch.device(parameters.device)
    model.to(device)
    device_count = 0 if device.type == "cpu" else torch.cuda.device_count()

    # Setup checkpointing
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

    # Initialize trainer
    trainer = Trainer(
        default_root_dir=os.path.join(EMOTION_MAPPING_PATH, parameters.dataset_name, parameters.training_name),
        devices=device_count,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        profiler="simple",
        callbacks=[checkpoint_callback, lr_monitor],
        enable_checkpointing=True,
        max_epochs=parameters.epochs,
        max_steps=parameters.max_training_steps if parameters.max_training_steps > 0 else None,
        log_every_n_steps=100,
        val_check_interval=500,
        limit_val_batches=64,
        num_sanity_val_steps=0,
    )

    # Train model
    trainer.fit(model, datamodule=datamodule)

    return {"Message": "Trained Emotion Mapper"}


def generate_bar_emotions(parameters: EmotionMapperGenerateParameters) -> dict:
    """Generate bar-level emotions for pieces"""

    processed_dir = os.path.join(PROCESSED_PATH, parameters.dataset_name)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    print(f"Saving generated emotions to processed files in: {processed_dir}")

    # Load model from checkpoint
    emotion_mapper_checkpoint = os.path.join(
        CHECKPOINTS_PATH,
        parameters.dataset_name,
        parameters.training_name,
        parameters.checkpoint_name,
    )
    model = load_emotion_mapper_from_checkpoint(emotion_mapper_checkpoint)

    # Load datamodule parameters
    with open("shared/assets/config.json", "r") as f:
        config = json.load(f)

    datamodule_parameters = config["dataloader"]
    datamodule_parameters["context_size"] = -1

    # Initialize datamodule
    datamodule = DataloaderModule(**datamodule_parameters)

    datamodule.setup("test")
    midi_files = datamodule.predict_ds.files
    random.shuffle(midi_files)

    datamodule_parameters["vocab"] = RemiVocab()
    dataset = DataloaderDataset(midi_files, **datamodule_parameters)

    coll = SeqCollator(context_size=-1)
    dataloader = DataLoader(dataset, batch_size=datamodule_parameters["batch_size"], collate_fn=coll)

    # Track statistics
    processed = 0
    successful = 0
    errors = []

    # Generate bar-level emotions for each batch
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Generate bar-level emotions
            bar_emotions = model.generate_bar_level_emotions(batch)

            # Save results
            for i, file in enumerate(bar_emotions):
                try:
                    # Load existing processed data
                    processed_data = async_load(processed_dir, file, "processed")

                    # Skip if emotions already exist
                    if "emotions" in processed_data:
                        continue

                    # Update processed data with emotions
                    processed_data["emotions"] = {
                        "bar_emotion_vectors": bar_emotions[file]["predictions"],
                        "piece_emotion_vector": bar_emotions[file]["emotions_vector"],
                        "bar_emotion_tokens": list(
                            chain.from_iterable(convert_emotions_to_sequence(emotions, idx) for idx, emotions in enumerate(bar_emotions[file]["predictions"]))
                        ),
                        "piece_emotion_tokens": convert_emotions_to_sequence(bar_emotions[file]["emotions_vector"]),
                        "avg_predictions": bar_emotions[file]["avg_predictions"],
                    }

                    # Save back to processed file
                    save_async(processed_dir, file, processed_data, "processed")
                    successful += 1

                except Exception as e:
                    errors.append(f"Error processing {os.path.basename(file)}: {str(e)}")

                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed} files", flush=True)

    return {"Message": "Generated Bar-level Emotions", "Total Files": processed, "Successful": successful, "Failed": len(errors), "Errors": errors}
