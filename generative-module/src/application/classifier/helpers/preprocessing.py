import os
import traceback
from pathlib import Path
import torchaudio
import pandas as pd
import torch
from tqdm import tqdm
import pickle
from application.classifier.helpers.midi_helper.remi.midi2event import analyzer, corpus, event


def torch_sox_effect_load(mp3_path, resample_rate):
    effects = [["rate", str(resample_rate)]]
    waveform, source_sr = torchaudio.load(mp3_path)
    if source_sr != 22050:
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, source_sr, effects, channels_first=True)
    return waveform


def remi_extractor(midi_path, event_to_int):
    midi_obj = analyzer(midi_path)
    song_data = corpus(midi_obj)
    event_sequence = event(song_data)
    quantize_midi = [event_to_int[str(i["name"]) + "_" + str(i["value"])] for i in event_sequence]
    return quantize_midi


def midi_feature_extract(
    midi_path="./datasets/ReMIDICaps/midi",
    remi_path="./datasets/ReMIDICaps/remi_midi",
    csv_path="./datasets/ReMIDICaps/labels/ReMIDICaps.csv",
    dictionary_path="datasets/ReMIDICaps/dictionary.pkl",
):
    """
    Extract REMI features from MIDI files.

    Args:
        midi_path: Path to the directory containing MIDI files
        remi_path: Path where extracted REMI features will be saved
        csv_path: Path to the CSV file containing the list of MIDI files to process
        dictionary_path: Path to the dictionary pickle file for REMI extraction

    Returns:
        int: The number of successfully processed MIDI files
    """
    # load remi dictionary
    path_dictionary = dictionary_path
    midi_dictionary = pickle.load(open(path_dictionary, "rb"))
    event_to_int = midi_dictionary[0]

    # Load CSV file containing the list of midi files to process
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Assuming the CSV has a column with midi filenames
        midi_files_in_csv = set(df["file"].values)  # Adjust column name as needed
        # Filter midi files that are in the CSV
        midi_files_to_process = [midi for midi in os.listdir(midi_path) if midi in midi_files_in_csv]
    else:
        print(f"CSV file not found at {csv_path}, processing all midi files")
        midi_files_to_process = os.listdir(midi_path)

    for midi in midi_files_to_process:
        remi_fn = os.path.join(remi_path, midi).replace(".mid", ".pt")
        try:
            remi_midi = remi_extractor(os.path.join(midi_path, midi), event_to_int)
        except:
            print(traceback.print_exc())
            return {"Status": "Fail"}
        if not os.path.exists(os.path.dirname(remi_fn)):
            os.makedirs(os.path.dirname(remi_fn))
        torch.save(remi_midi, remi_fn)


if __name__ == "__main__":
    midi_feature_extract()
    # audio_domain_resample()
