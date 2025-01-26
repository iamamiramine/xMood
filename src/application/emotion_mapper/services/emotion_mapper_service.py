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
    LABELS_PATH,
    PROCESSED_PATH,
)

# Domain models
from domain.models.emotion_mapper.emotion_mapper_model import (
    EmotionMapperTrainingParameters,
    EmotionMapperGenerateParameters,
)


def preprocess_dataset_emotions(dataset_name: str, save: bool = True) -> dict:
    """Preprocess emotion vectors into sequences for an entire dataset.

    Args:
        dataset_name: Name of the dataset to process
        split: Optional split name (train/val/test)
        save: Whether to save the processed sequences

    Returns:
        Dictionary containing the processing summary
    """
    # Create output directory if saving
    if save:
        out_dir = os.path.join(LABELS_PATH, dataset_name, "emotion_sequences")
        os.makedirs(out_dir, exist_ok=True)

    # Read emotion labels
    labels_df, label_columns = read_labels(dataset_name)
    emotion_vocab = EmotionVocab()

    processed = 0
    errors = []

    # Process each file in the dataset
    for idx, row in labels_df.iterrows():
        try:
            file_name = row["file_name"]
            emotions = row[label_columns].values

            # Create sequences for each bar (assuming max 512 bars per piece)
            emotion_tokens = convert_emotions_to_sequence(emotions)

            # Encode tokens
            encoded_emotions = emotion_vocab.encode(emotion_tokens)

            result = {
                "file": file_name,
                "emotion_tokens": emotion_tokens,
                "encoded_emotions": encoded_emotions,
                "decoded_emotions": emotion_vocab.decode(encoded_emotions),
                "emotions_vector": emotions,
            }

            if save:
                output_file = os.path.join(out_dir, f"{file_name}_emotions.pkl")
                with open(output_file, "wb") as f:
                    pickle.dump(result, f)

            processed += 1

        except Exception as e:
            errors.append(f"Error processing {file_name}: {str(e)}")

    return {"Message": "Dataset Emotion Sequence Processing Complete"}


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
