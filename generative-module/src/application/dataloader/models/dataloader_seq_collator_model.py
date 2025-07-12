import torch
from torch.nn.utils.rnn import pad_sequence
from domain.constants.model_constants import ModelConstants


class SeqCollator:
    def __init__(self, pad_token=ModelConstants.PAD_TOKEN_ID, context_size=ModelConstants.DEFAULT_CONTEXT_SIZE, device=None):
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

        if "moods" in features[0]:
            moods = [feature["moods"] for feature in features]
            moods = pad_sequence(moods, batch_first=True, padding_value=self.pad_token)
            batch["moods"] = moods

        # Handle global features if present
        if "global_features" in features[0]:
            global_features = [feature["global_features"] for feature in features]
            # Check if they're tensors or need to be converted
            if not isinstance(global_features[0], torch.Tensor):
                global_features = [torch.tensor(gf, dtype=torch.float) if not isinstance(gf, torch.Tensor) else gf 
                                for gf in global_features]
            # Use pad_sequence only if the global features have sequence dimension
            if len(global_features[0].shape) > 1:
                global_features = pad_sequence(global_features, batch_first=True, padding_value=0.0)
            else:
                # If they're just feature vectors (no sequence), stack them
                global_features = torch.stack(global_features)
            batch["global_features"] = global_features

        # Handle text prompts if present
        if "text_prompts" in features[0]:
            # Just collect the text strings, they'll be tokenized later
            batch["text_prompts"] = [feature["text_prompts"] for feature in features]

        if "file" in features[0]:
            batch["files"] = [feature["file"] for feature in features]

        return batch
