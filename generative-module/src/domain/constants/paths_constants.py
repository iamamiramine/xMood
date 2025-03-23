import json
import os

# with open(os.path.join("shared", "assets", "paths_demo.json")) as file:
with open(os.path.join("shared", "assets", "paths.json")) as file:
    paths = json.load(file)

with open(os.path.join("shared", "assets", "config.json")) as file:
    config = json.load(file)

ROOT_OUTPUT = paths["ROOT_OUTPUT"]
DATASETS_PATH = paths["DATASETS_PATH"]

DATASET_NAME = config["dataloader"]["dataset_name"]

MIDI_PATH = f"{DATASETS_PATH}/{DATASET_NAME}/midi/"
LABELS_PATH = f"{DATASETS_PATH}/{DATASET_NAME}/labels/"

PROCESSED_PATH = f"{ROOT_OUTPUT}/processed/"
CHECKPOINTS_PATH = f"{ROOT_OUTPUT}/checkpoints/"
GENERATED_PATH = f"{ROOT_OUTPUT}/generated/"
LATENTS_PATH = f"{ROOT_OUTPUT}/latents/"


CHORDS_PATH = f"{ROOT_OUTPUT}/chords/"
KEYS_PATH = f"{ROOT_OUTPUT}/keys/"

ENCODINGS_PATH = f"{ROOT_OUTPUT}/encodings/"

SYMBOLIC_FEATURES_PATH = f"{ROOT_OUTPUT}/symbolic_features/"

REPRESENTATIONS_PATH = f"{ROOT_OUTPUT}/representations/"

VAE_PATH = f"{ROOT_OUTPUT}/VQVAE/"
CODES_PATH = f"{ROOT_OUTPUT}/codes/"
FEATURES_PATH = f"{ROOT_OUTPUT}/features/"

EMOTION_MAPPING_PATH = f"{ROOT_OUTPUT}/emotion_mapping/"

GENERATOR_PATH = f"{ROOT_OUTPUT}/generator/"
