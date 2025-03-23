import pickle
import asyncio
import os
import aiofiles
import io

import pretty_midi as pm

import torch


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


# Asynchronous function to save the tonal plan
def save_async(out_dir, midi_file, data, file_name):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, f"{os.path.basename(midi_file)}_{file_name}.pkl"), "wb") as f:
        pickle.dump(data, f)


def async_load(out_dir, midi_file, file_name):
    file_path = os.path.join(out_dir, f"{os.path.basename(midi_file)}_{file_name}.pkl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data
