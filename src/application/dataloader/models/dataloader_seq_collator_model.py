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
            batch["bar_symbolic"] = bar_symbolic

        if "bar_symbolic_ids" in features[0]:
            bar_symbolic_ids = [feature["bar_symbolic_ids"] for feature in features]
            bar_symbolic_ids = pad_sequence(bar_symbolic_ids, batch_first=True, padding_value=self.pad_token)
            bar_symbolic_ids = bar_symbolic_ids[:, :max_symb_len]
            batch["bar_symbolic_ids"] = bar_symbolic_ids
            bar_symb_len = bar_symbolic_ids.size(1)

            if "symb_bar_ids" in features[0]:
                symb_bar_ids = [feature["symb_bar_ids"] for feature in features]
                symb_bar_ids = pad_sequence(symb_bar_ids, batch_first=True, padding_value=0)
                batch["symb_bar_ids"] = symb_bar_ids[:, :bar_symb_len]

            if "symb_position_ids" in features[0]:
                symb_position_ids = [feature["symb_position_ids"] for feature in features]
                symb_position_ids = pad_sequence(symb_position_ids, batch_first=True, padding_value=0)
                batch["symb_position_ids"] = symb_position_ids[:, :bar_symb_len]

        if "piece_symbolic" in features[0]:
            piece_symbolic = [feature["piece_symbolic"] for feature in features]
            batch["piece_symbolic"] = piece_symbolic

        if "piece_symbolic_ids" in features[0]:
            piece_symbolic_ids = [feature["piece_symbolic_ids"] for feature in features]
            piece_symbolic_ids = pad_sequence(piece_symbolic_ids, batch_first=True, padding_value=self.pad_token)
            piece_symbolic_ids = piece_symbolic_ids[:, :max_symb_len]
            batch["piece_symbolic_ids"] = piece_symbolic_ids

        if "piece_emotions_tokens" in features[0]:
            piece_emotions_tokens = [feature["piece_emotions_tokens"] for feature in features]
            batch["piece_emotions_tokens"] = piece_emotions_tokens

        if "piece_emotions_ids" in features[0]:
            piece_emotions_ids = [feature["piece_emotions_ids"] for feature in features]
            piece_emotions_ids = pad_sequence(piece_emotions_ids, batch_first=True, padding_value=self.pad_token)
            batch["piece_emotions_ids"] = piece_emotions_ids

        if "file" in features[0]:
            batch["files"] = [feature["file"] for feature in features]

        return batch
