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
            max_desc_len = self.context_size
        else:
            max_len = xs.size(1)
            max_desc_len = int(1e4)

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
            batch["latents"] = latents[:, :max_desc_len]

        if "codes" in features[0]:
            codes = [feature["codes"] for feature in features]
            codes = pad_sequence(codes, batch_first=True, padding_value=0)
            batch["codes"] = codes[:, :max_desc_len]

        if "description" in features[0]:
            description = [feature["description"] for feature in features]
            description = pad_sequence(description, batch_first=True, padding_value=self.pad_token)
            desc = description[:, :max_desc_len]
            batch["description"] = desc

            if "desc_bar_ids" in features[0]:
                desc_len = desc.size(1)
                desc_bar_ids = [feature["desc_bar_ids"] for feature in features]
                desc_bar_ids = pad_sequence(desc_bar_ids, batch_first=True, padding_value=0)
                batch["desc_bar_ids"] = desc_bar_ids[:, :desc_len]

        if "sentiment_vector" in features[0]:
            sentiment_vector = [feature["sentiment_vector"] for feature in features]
            sentiment_vector = torch.tensor(sentiment_vector, dtype=torch.float32, device=self.device)
            sentiment_vector = sentiment_vector.expand(batch_size, -1)
            batch["sentiment_vector"] = sentiment_vector

        if "file" in features[0]:
            batch["files"] = [feature["file"] for feature in features]

        return batch
