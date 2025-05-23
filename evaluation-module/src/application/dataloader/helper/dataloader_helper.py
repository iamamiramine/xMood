import torch

from persistence.dataloader.repositories.dataloader_repository import (
    save_async,
)
from application.encoder.helpers.vocab_helper import (
    get_positions,
    get_bars,
    mask_bar_tokens,
    get_bos_eos_events,
)
from application.encoder.models.vocab_model import RemiVocab, SymbolicFeaturesVocab
from domain.constants.encoder.token_constants import (
    BAR_KEY,
    BOS_TOKEN,
    EOS_TOKEN,
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
    bar_symbolic=None,
    save=False,
    out_dir="",
    api_call=False,
    processed_data=None,
    use_symb_pos=False,
):
    vocab = RemiVocab()
    symb_vocab = SymbolicFeaturesVocab()
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

        if bar_symbolic is not None:
            min_bar = b_ids[0]
            symb_events = bar_symbolic
            # Remove Position keys if use_symb_pos is False
            if not use_symb_pos:
                symb_events = [event for event in symb_events if not event.startswith("Position_")]

            symb_bars = [i for i, event in enumerate(symb_events) if f"{BAR_KEY}_" in event]
            start_idx = symb_bars[max(0, min_bar - 1)]

            symb_bar_ids = torch.zeros(len(symb_events), dtype=torch.int)
            symb_bar_ids[symb_bars] = 1
            symb_bar_ids = torch.cumsum(symb_bar_ids, dim=0)

            symb_position_ids = None
            if use_symb_pos:
                symb_position_ids = get_positions(symb_events)
                symb_position_ids = torch.cat([zero, symb_position_ids, zero])

            if max_bars_per_context and max_bars_per_context > 0:
                end_idx = symb_bars[min_bar + max_bars_per_context]
                symb_events = symb_events[start_idx:end_idx]
                symb_bar_ids = symb_bar_ids[start_idx:end_idx]
                if use_symb_pos:
                    symb_position_ids = symb_position_ids[start_idx:end_idx]
                start_idx = 0

            symb_bos = torch.tensor(symb_vocab.encode([BOS_TOKEN]), dtype=torch.int)
            symb_eos = torch.tensor(symb_vocab.encode([EOS_TOKEN]), dtype=torch.int)
            symb_ids = torch.tensor(symb_vocab.encode(symb_events), dtype=torch.int)

            if min_bar == 0:
                symb_ids = torch.cat([symb_bos, symb_ids, symb_eos])
                symb_bar_ids = torch.cat([zero, symb_bar_ids, zero])
            else:
                symb_ids = torch.cat([symb_ids, symb_eos])
                symb_bar_ids = torch.cat([symb_bar_ids, zero])

            if context_size > 0:
                start, end = start_idx, start_idx + context_size + 1
                x["bar_symbolic"] = symb_ids[start:end]
                x["symb_bar_ids"] = symb_bar_ids[start:end]
                if use_symb_pos:
                    x["symb_position_ids"] = symb_position_ids[start:end]
            else:
                x["bar_symbolic"] = symb_ids[start:]
                x["symb_bar_ids"] = symb_bar_ids[start:]
                if use_symb_pos:
                    x["symb_position_ids"] = symb_position_ids[start:]

        if latents is not None:
            x["latents"] = latents
            x["codes"] = codes

        if save:
            if processed_data is not None:
                # Update the processed file with the representation
                processed_data["representation"] = x
                save_async(out_dir, midi, processed_data, "processed")
            else:
                # Fallback to old behavior if processed_data not provided
                save_async(out_dir, midi, x, "processed")

        return x if not api_call else {"Message": "Represented Encoding"}
