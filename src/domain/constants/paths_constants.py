import json
import os

with open(os.path.join("shared", "assets", "paths.json")) as file:
    paths = json.load(file)

ROOT_OUTPUT = paths["ROOT_OUTPUT"]
DATASETS_PATH = paths["DATASETS_PATH"]

MIDI_PATH = f"{DATASETS_PATH}/midi/"
LABELS_PATH = f"{DATASETS_PATH}/labels/"

DATALOADER_PATH = f"{ROOT_OUTPUT}/dataloader/"

CHORDS_PATH = f"{ROOT_OUTPUT}/chords/"
KEYS_PATH = f"{ROOT_OUTPUT}/keys/"

ENCODINGS_PATH = f"{ROOT_OUTPUT}/encodings/"

CHECKPOINTS_PATH = f"{ROOT_OUTPUT}/checkpoints/"

DESCRIPTIONS_PATH = f"{ROOT_OUTPUT}/descriptions/"

REPRESENTATIONS_PATH = f"{ROOT_OUTPUT}/representations/"

VAE_PATH = f"{ROOT_OUTPUT}/VQVAE/"
LATENTS_PATH = f"{ROOT_OUTPUT}/latents/"
CODES_PATH = f"{ROOT_OUTPUT}/codes/"
FEATURES_PATH = f"{ROOT_OUTPUT}/features/"

SENTIMENT_LEARNER_PATH = f"{ROOT_OUTPUT}/sentiment_learner/"
ANN_PATH = f"{ROOT_OUTPUT}/ann/"

GENERATOR_PATH = f"{ROOT_OUTPUT}/generator/"
GENERATED_PATH = f"{ROOT_OUTPUT}/generated/"
