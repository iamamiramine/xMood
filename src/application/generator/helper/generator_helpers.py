import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.models.bert.modeling_bert import BertAttention

from src.application.encoder.helpers.remi_helper import remi2midi
from src.application.generator.models.generator_model import MIDIGeneratorModule


def load_generator_from_checkpoint(checkpoint_dir: str, eval=True):
    pl_ckpt = torch.load(checkpoint_dir, map_location="cpu")
    kwargs = pl_ckpt["hyper_parameters"]
    kwargs["save_encoder_decoder_path"] = "output/experiments/generator_remi_code/generator/5_Emotions_MMD_encoded/generator_remi_code_training/BERT"
    kwargs["load_bert_from_ckpt"] = True
    model = MIDIGeneratorModule(**kwargs)
    state_dict = pl_ckpt["state_dict"]
    # position_ids are no longer saved in the state_dict starting with transformers==4.31.0
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith("embeddings.position_ids")}
    try:
        # succeeds for checkpoints trained with transformers>4.13.0
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        config = model.transformer.decoder.bert.config
        for layer in model.transformer.decoder.bert.encoder.layer:
            layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
        model.load_state_dict(state_dict)
    if eval:
        model.freeze()
        model.eval()
    return model


def combine_batches(batches, bars_per_sequence=8, description_flavor="none", device=None):
    if device is None:
        device = batches[0]["input_ids"].device

    batch_size = batches[0]["input_ids"].size(0)

    zero = torch.zeros(1, device=device, dtype=torch.int)

    contexts = []
    batch_ = {}

    for i in range(batch_size):
        curr_bar = 0
        ctx = {
            "input_ids": [],
            "bar_ids": [],
            "position_ids": [],
            "slices": [],
            "description": [],
            "desc_bar_ids": [],
            "desc_slices": [],
            "latents": [],
            "latent_slices": [],
            "files": [],
        }

        for batch in batches:
            if i >= batch["input_ids"].size(0):
                continue

            curr = curr_bar

            bar_ids = batch["bar_ids"][i]
            starts = (bar_ids >= curr).nonzero()
            ends = (bar_ids >= max(1, curr) + bars_per_sequence).nonzero()
            if starts.size(0) == 0:
                continue
            start = starts[0, 0]

            if ends.size(0) == 0:
                end = bar_ids.size(0)
                curr_bar = bar_ids[-1] + 1
            else:
                end = ends[0, 0]
                curr_bar = bar_ids[end]

            if description_flavor in ["description", "both"]:
                desc_bar_ids = batch["desc_bar_ids"][i]
                desc_start = (desc_bar_ids >= curr).nonzero()[0, 0]
                desc_ends = (desc_bar_ids >= max(1, curr) + bars_per_sequence).nonzero()

                if desc_ends.size(0) == 0:
                    desc_end = desc_bar_ids.size(0)
                else:
                    desc_end = desc_ends[0, 0]

            if description_flavor in ["latent", "both"]:
                latent_start = curr
                latent_end = max(1, curr) + bars_per_sequence

            ctx["input_ids"].append(batch["input_ids"][i, start:end])
            ctx["bar_ids"].append(batch["bar_ids"][i, start:end])
            ctx["position_ids"].append(batch["position_ids"][i, start:end])
            ctx["slices"].append((start, end))
            if description_flavor in ["description", "both"]:
                ctx["description"].append(batch["description"][i, desc_start:desc_end])
                ctx["desc_bar_ids"].append(batch["desc_bar_ids"][i, desc_start:desc_end])
                ctx["desc_slices"].append((desc_start, desc_end))
            if description_flavor in ["latent", "both"]:
                ctx["latents"].append(batch["latents"][i, latent_start:latent_end])
                ctx["latent_slices"].append((latent_start, latent_end))
            ctx["files"].append(batch["files"][i])

        if len(ctx["files"]) <= 1:
            continue

        keys = ["input_ids", "bar_ids", "position_ids", "description", "desc_bar_ids", "latents"]
        for key in keys:
            if key in ctx and len(ctx[key]) > 0:
                ctx[key] = torch.cat(ctx[key])
        ctx["labels"] = torch.cat([ctx["input_ids"][1:], zero])
        ctx["files"] = "__".join(ctx["files"]).replace(".mid", "") + ".mid"

        contexts.append(ctx)

    batch_["files"] = [ctx["files"] for ctx in contexts]

    for key in ["input_ids", "bar_ids", "position_ids", "description", "desc_bar_ids", "latents", "labels"]:
        xs = [ctx[key] for ctx in contexts if isinstance(ctx[key], torch.Tensor)]
        if len(xs) > 0:
            xs = pad_sequence(xs, batch_first=True, padding_value=0)
            if not key in ["latents"]:
                xs = xs.long()
            batch_[key] = xs

    return batch_


def medley_iterator(dl, n_pieces=2, n_bars=8, description_flavor="none"):
    dl_iter = iter(dl)
    try:
        while True:
            batches = [next(dl_iter) for _ in range(n_pieces)]
            batch = combine_batches(batches, bars_per_sequence=n_bars, description_flavor=description_flavor)
            yield batch
    except StopIteration:
        return


@torch.no_grad()
def reconstruct_sample(
    model,
    batch,
    initial_context=1,
    output_dir=None,
    max_iter=-1,
    max_bars=-1,
    verbose=0,
    description_flavor="none",
    prompt_name="prompt",
    sequence_length=1024,
):
    batch_size, seq_len = batch["input_ids"].shape[:2]

    batch_ = {key: batch[key][:, :initial_context] for key in ["input_ids", "bar_ids", "position_ids"]}
    if description_flavor in ["description", "both"]:
        batch_["description"] = batch["description"]
        batch_["desc_bar_ids"] = batch["desc_bar_ids"]
    if description_flavor in ["latent", "both"]:
        batch_["latents"] = batch["latents"]

    max_len = sequence_length  # + 1024
    if max_iter > 0:
        max_len = min(max_len, initial_context + max_iter)
    if verbose:
        print(f"Generating sequence ({initial_context} initial / {max_len} max length / {max_bars} max bars / {batch_size} batch size)", flush=True)
    sample = model.sample(batch_, max_length=max_len, max_bars=max_bars, verbose=verbose)

    xs = batch["input_ids"].detach().cpu()
    xs_hat = sample["sequences"].detach().cpu()  # Generated
    events = [model.vocab.decode(x) for x in xs]
    events_hat = [model.vocab.decode(x) for x in xs_hat]  # Generated

    pms, pms_hat = [], []
    n_fatal = 0
    for rec, rec_hat in zip(events, events_hat):
        try:
            pm = remi2midi(rec)
            pms.append(pm)
        except Exception as err:
            print("ERROR: Could not convert events to midi:", err)
        try:
            pm_hat = remi2midi(rec_hat)  # Generated
            pms_hat.append(pm_hat)
        except Exception as err:
            print("ERROR: Could not convert events to midi:", err)
            n_fatal += 1

    if output_dir:
        os.makedirs(os.path.join(output_dir, "ground_truth"), exist_ok=True)
        for pm, pm_hat, file in zip(pms, pms_hat, batch["files"]):
            if verbose:
                print(f"Saving to {output_dir}/{file}")
            pm.write(os.path.join(output_dir, "ground_truth", file))
            pm_hat.write(os.path.join(output_dir, f"{prompt_name}_{file}"))  # Generated

    return events


# Function to compute the Euclidean distance between two sets of labels
def euclidean_distance(dict1, dict2):
    keys = dict1.keys()
    distance = np.sqrt(sum((dict1[key] - dict2[key]) ** 2 for key in keys))
    return distance


# Function to find the file with the closest labels to the target
def find_closest_file(target_labels, file_labels):
    closest_file = None
    min_distance = float("inf")

    for file, labels in file_labels.items():
        distance = euclidean_distance(target_labels, labels)
        if distance < min_distance:
            min_distance = distance
            closest_file = file

    return closest_file, min_distance
