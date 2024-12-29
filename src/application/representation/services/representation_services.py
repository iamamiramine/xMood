import os
import torch

from src.application.feature_extraction.helpers.latent_features_helper import read_label_for_midi
from src.persistence.dataloader.repositories.dataloader_repository import (
    save_async,
    async_load,
)
from src.domain.constants.paths_constants import (
    DESCRIPTIONS_PATH,
    LATENTS_PATH,
    ENCODINGS_PATH,
)
from src.domain.models.representation.representation_model import (
    RepresentationParameters,
)
from src.persistence.dataloader.repositories.dataloader_repository import CPU_Unpickler
from src.application.encoder.helpers.vocab_helper import (
    get_positions,
    get_bars,
    mask_bar_tokens,
    get_bos_eos_events,
)
from src.application.encoder.models.vocab_model import RemiVocab, DescriptionVocab
from src.domain.constants.encoder.token_constants import (
    BAR_KEY,
    BOS_TOKEN,
    EOS_TOKEN,
)


def run_representation(parameters: RepresentationParameters):
    try:
        encoding_dir = os.path.join(
            str(ENCODINGS_PATH),
            parameters.dataset_name,
        )
        encoding = async_load(encoding_dir, parameters.midi, "encoding")
        events = encoding["events"]
    except ValueError as err:
        print(err)

    if parameters.load_desc:
        try:
            desc_dir = os.path.join(
                str(DESCRIPTIONS_PATH),
                parameters.dataset_name,
            )
            desc = async_load(desc_dir, parameters.midi, "description")
            description = desc["description"]
        except ValueError as err:
            print(err)
    else:
        description = None

    if parameters.load_latent:
        try:
            latents_path = os.path.join(
                str(LATENTS_PATH),
                parameters.dataset_name,
                f"{os.path.basename((parameters.midi))}_latents.pkl",
            )
            with open(latents_path, "rb") as f:
                latents_file = CPU_Unpickler(f).load()

            latents = latents_file["latents"]
            codes = latents_file["codes"]
        except ValueError as err:
            print(err)
    else:
        latents = None
        codes = None

    return represent_encoding(
        parameters.midi,
        parameters.dataset_name,
        parameters.context_size,
        parameters.max_bars,
        parameters.max_positions,
        parameters.bar_token_mask,
        parameters.max_bars_per_context,
        parameters.max_contexts_per_file,
        events=events,
        latents=latents,
        codes=codes,
        description=description,
        save=parameters.save,
        out_dir=parameters.out_dir,
        api_call=parameters.api_call,
    )


def represent_encoding(
    midi,
    dataset_name,
    context_size,
    max_bars,
    max_positions,
    bar_token_mask,
    max_bars_per_context,
    max_contexts_per_file,
    events,
    latents=None,
    codes=None,
    description=None,
    save=False,
    out_dir="",
    api_call=False,
):
    vocab = RemiVocab()
    bars, bar_ids = get_bars(events, include_ids=True)
    if len(bars) > max_bars:
        print(f"WARNING: REMI sequence has more than {max_bars} bars: {len(bars)} event bars.")

    position_ids = get_positions(events)
    max_pos = position_ids.max()
    if max_pos > max_positions:
        print(f"WARNING: REMI sequence has more than {max_positions} positions: {max_pos.item()} positions found")

    if bar_token_mask is not None and max_bars_per_context > 0:
        events = mask_bar_tokens(events, bar_token_mask=bar_token_mask)

    event_ids = torch.tensor(vocab.encode(events), dtype=torch.long)

    bos, eos = get_bos_eos_events(vocab)
    zero = torch.tensor([0], dtype=torch.int)

    if max_bars_per_context and max_bars_per_context > 0:
        starts = [bars[i] for i in range(0, len(bars), max_bars_per_context)]
        contexts = list(zip(starts[:-1], starts[1:])) + [(starts[-1], len(event_ids))]
    else:
        event_ids = torch.cat([bos, event_ids, eos])
        bar_ids = torch.cat([zero, bar_ids, zero])
        position_ids = torch.cat([zero, position_ids, zero])

        if context_size > 0:
            starts = list(range(0, len(event_ids), context_size + 1))
            if len(starts) > 1:
                contexts = [(start, start + context_size + 1) for start in starts[:-1]] + [(len(event_ids) - (context_size + 1), len(event_ids))]
            elif len(starts) > 0:
                contexts = [(starts[0], context_size + 1)]
        else:
            contexts = [(0, len(event_ids))]

    if max_contexts_per_file and max_contexts_per_file > 0:
        contexts = contexts[:max_contexts_per_file]

    for start, end in contexts:
        if max_bars_per_context and max_bars_per_context > 0:
            src = torch.cat([bos, event_ids[start:end], eos])
            b_ids = torch.cat([zero, bar_ids[start:end], zero])
            p_ids = torch.cat([zero, position_ids[start:end], zero])
        else:
            src = event_ids[start:end]
            b_ids = bar_ids[start:end]
            p_ids = position_ids[start:end]

        if context_size > 0:
            src = src[: context_size + 1]

        x = {
            "file": midi,
            "input_ids": src,
            "bar_ids": b_ids,
            "position_ids": p_ids,
        }

        if description is not None:
            desc_vocab = DescriptionVocab()
            min_bar = b_ids[0]
            desc_events = description
            desc_bars = [i for i, event in enumerate(desc_events) if f"{BAR_KEY}_" in event]
            start_idx = desc_bars[max(0, min_bar - 1)]

            desc_bar_ids = torch.zeros(len(desc_events), dtype=torch.int)
            desc_bar_ids[desc_bars] = 1
            desc_bar_ids = torch.cumsum(desc_bar_ids, dim=0)

            if max_bars_per_context and max_bars_per_context > 0:
                end_idx = desc_bars[min_bar + max_bars_per_context]
                desc_events = desc_events[start_idx:end_idx]
                desc_bar_ids = desc_bar_ids[start_idx:end_idx]
                start_idx = 0

            desc_bos = torch.tensor(desc_vocab.encode([BOS_TOKEN]), dtype=torch.int)
            desc_eos = torch.tensor(desc_vocab.encode([EOS_TOKEN]), dtype=torch.int)
            desc_ids = torch.tensor(desc_vocab.encode(desc_events), dtype=torch.int)

            if min_bar == 0:
                desc_ids = torch.cat([desc_bos, desc_ids, desc_eos])
                desc_bar_ids = torch.cat([zero, desc_bar_ids, zero])
            else:
                desc_ids = torch.cat([desc_ids, desc_eos])
                desc_bar_ids = torch.cat([desc_bar_ids, zero])

            if context_size > 0:
                start, end = start_idx, start_idx + context_size + 1
                x["description"] = desc_ids[start:end]
                x["desc_bar_ids"] = desc_bar_ids[start:end]
            else:
                x["description"] = desc_ids[start:]
                x["desc_bar_ids"] = desc_bar_ids[start:]

        if latents is not None:
            x["latents"] = latents
            x["codes"] = codes

        # midi_labels_df, label_columns = read_label_for_midi(dataset_name, midi)
        # sentiment_vector = midi_labels_df[label_columns].values.flatten().tolist()
        # x["sentiment_vector"] = sentiment_vector

        if save:
            save_async(out_dir, midi, x, "_representation")

        return x if not api_call else {"Message": "Represented Encoding"}
