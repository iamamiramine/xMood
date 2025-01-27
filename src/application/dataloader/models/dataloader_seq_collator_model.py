import torch
from torch.nn.utils.rnn import pad_sequence


class SeqCollator:
    def __init__(self, pad_token=0, context_size=512, device=None):
        self.pad_token = pad_token
        self.context_size = context_size
        self.device = device

    def __call__(self, features):
        batch = {}

        # Handle input_ids if present
        if "input_ids" in features[0]:
            xs = [feature["input_ids"] for feature in features]
            xs = pad_sequence(xs, batch_first=True, padding_value=self.pad_token)

            if self.context_size > 0:
                max_len = self.context_size
                max_symb_len = self.context_size
            else:
                max_len = xs.size(1)
                max_symb_len = int(1e4)

            tmp = xs[:, : (max_len + 1)][:, :-1]
            labels = xs[:, : (max_len + 1)][:, 1:].clone().detach()
            xs = tmp

            seq_len = xs.size(1)

            batch["input_ids"] = xs
            batch["labels"] = labels

            # Get batch size from input_ids
            batch_size = batch["input_ids"].shape[0]
        else:
            # For captioning mode, get batch size from number of features
            batch_size = len(features)

        if "position_ids" in features[0]:
            position_ids = [feature["position_ids"] for feature in features]
            position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
            batch["position_ids"] = position_ids[:, :seq_len]

        if "bar_ids" in features[0]:
            bar_ids = [feature["bar_ids"] for feature in features]
            bar_ids = pad_sequence(bar_ids, batch_first=True, padding_value=0)
            batch["bar_ids"] = bar_ids[:, :seq_len]

        if "latents" in features[0]:
            latents = [feature["latents"] for feature in features]
            latents = pad_sequence(latents, batch_first=True, padding_value=0.0)
            batch["latents"] = latents[:, :max_symb_len]

        if "codes" in features[0]:
            codes = [feature["codes"] for feature in features]
            codes = pad_sequence(codes, batch_first=True, padding_value=0)
            batch["codes"] = codes[:, :max_symb_len]

        if "symbolic_ids" in features[0]:
            symbolic_ids = [feature["symbolic_ids"] for feature in features]
            symbolic_ids = pad_sequence(symbolic_ids, batch_first=True, padding_value=self.pad_token)
            symb_ids = symbolic_ids[:, :max_symb_len]
            batch["symbolic_ids"] = symb_ids

            if "symb_bar_ids" in features[0]:
                symb_len = symb_ids.size(1)
                symb_bar_ids = [feature["symb_bar_ids"] for feature in features]
                symb_bar_ids = pad_sequence(symb_bar_ids, batch_first=True, padding_value=0)
                batch["symb_bar_ids"] = symb_bar_ids[:, :symb_len]

        if "symbolic" in features[0]:
            symbolic = [feature["symbolic"] for feature in features]
            batch["symbolic"] = symbolic

        if "emotions_vector" in features[0]:
            emotions_vector = [feature["emotions_vector"] for feature in features]
            emotions_vector = torch.tensor(emotions_vector, dtype=torch.float32, device=self.device)
            emotions_vector = emotions_vector.expand(batch_size, -1)
            batch["emotions_vector"] = emotions_vector

        if "genre" in features[0]:
            batch["genre"] = [feature.get("genre", "") for feature in features]

        if "composer" in features[0]:
            batch["composer"] = [feature.get("composer", "") for feature in features]

        if "note_density" in features[0]:
            batch["note_density"] = [feature.get("note_density", "") for feature in features]

        if "mean_velocity" in features[0]:
            batch["mean_velocity"] = [feature.get("mean_velocity", "") for feature in features]

        if "mean_pitch" in features[0]:
            batch["mean_pitch"] = [feature.get("mean_pitch", "") for feature in features]

        if "mean_duration" in features[0]:
            batch["mean_duration"] = [feature.get("mean_duration", "") for feature in features]

        if "time_signatures" in features[0]:
            batch["time_signatures"] = [feature.get("time_signatures", "") for feature in features]

        if "key_signatures" in features[0]:
            batch["key_signatures"] = [feature.get("key_signatures", "") for feature in features]

        if "chords" in features[0]:
            batch["chords"] = [feature.get("chords", "") for feature in features]

        if "instruments" in features[0]:
            batch["instruments"] = [feature.get("instruments", "") for feature in features]

        if "emotions" in features[0]:
            batch["emotions"] = [feature.get("emotions", "") for feature in features]

        if "file" in features[0]:
            batch["files"] = [feature["file"] for feature in features]

        return batch
