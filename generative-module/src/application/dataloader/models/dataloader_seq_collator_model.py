import torch
from torch.nn.utils.rnn import pad_sequence


class SeqCollator:
    def __init__(self, pad_token=0, context_size=512, device=None):
        self.pad_token = pad_token
        self.context_size = context_size
        self.device = device

    def __call__(self, features):
        batch = {}

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

        if "bar_symbolic" in features[0]:
            bar_symbolic = [feature["bar_symbolic"] for feature in features]
            bar_symbolic = pad_sequence(bar_symbolic, batch_first=True, padding_value=self.pad_token)
            bar_symb = bar_symbolic[:, :max_symb_len]
            batch["bar_symbolic"] = bar_symb
            symb_len = bar_symb.size(1)

            if "symb_bar_ids" in features[0]:
                symb_bar_ids = [feature["symb_bar_ids"] for feature in features]
                symb_bar_ids = pad_sequence(symb_bar_ids, batch_first=True, padding_value=0)
                batch["symb_bar_ids"] = symb_bar_ids[:, :symb_len]

            if "symb_position_ids" in features[0]:
                symb_position_ids = [feature["symb_position_ids"] for feature in features]
                symb_position_ids = pad_sequence(symb_position_ids, batch_first=True, padding_value=0)
                batch["symb_position_ids"] = symb_position_ids[:, :symb_len]

        if "piece_symbolic" in features[0]:
            piece_symbolic = [feature["piece_symbolic"] for feature in features]
            piece_symbolic = pad_sequence(piece_symbolic, batch_first=True, padding_value=self.pad_token)
            piece_symb = piece_symbolic[:, :max_symb_len]
            batch["piece_symbolic"] = piece_symb

        if "moods" in features[0]:
            moods = [feature["moods"] for feature in features]
            moods = pad_sequence(moods, batch_first=True, padding_value=self.pad_token)
            batch["moods"] = moods

        if "file" in features[0]:
            batch["files"] = [feature["file"] for feature in features]

        return batch
