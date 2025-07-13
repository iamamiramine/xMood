import os
import h5py
import numpy as np
import torch

from application.symbolic.helpers.vocab_helper import (
    get_positions,
    get_bars,
    mask_bar_tokens,
    get_bos_eos_events,
)
from core.symbolic.models.vocab_model import RemiVocab, SymbolicFeaturesVocab
from domain.constants.token_constants import (
    BAR_KEY,
    BOS_TOKEN,
    EOS_TOKEN,
)


def _tensor_to_numpy(tensor):
    """Convert PyTorch tensor to numpy array for HDF5 storage."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def _numpy_to_tensor(array):
    """Convert numpy array back to PyTorch tensor."""
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    return array


def _save_to_hdf5(data, filepath):
    """Save data to HDF5 file with proper handling of different data types."""
    with h5py.File(filepath, 'w') as f:
        def _save_recursive(group, key, value):
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                for k, v in value.items():
                    _save_recursive(subgroup, k, v)
            elif isinstance(value, list):
                # Handle lists by converting to numpy arrays if possible
                if all(isinstance(item, (int, float, str)) for item in value):
                    group.create_dataset(key, data=np.array(value))
                elif all(isinstance(item, torch.Tensor) for item in value):
                    # Convert list of tensors to numpy arrays
                    tensor_data = [_tensor_to_numpy(item) for item in value]
                    # Create a group for the list of tensors
                    list_group = group.create_group(key)
                    for i, tensor in enumerate(tensor_data):
                        list_group.create_dataset(f'item_{i}', data=tensor)
                    list_group.attrs['is_tensor_list'] = True
                else:
                    # For mixed lists, convert to string representation
                    str_data = [str(item) for item in value]
                    group.create_dataset(key, data=np.array(str_data, dtype='S'))
                    group[key].attrs['is_string_list'] = True
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                group.create_dataset(key, data=_tensor_to_numpy(value))
                if isinstance(value, torch.Tensor):
                    group[key].attrs['is_tensor'] = True
            elif isinstance(value, (int, float, str, bool)):
                group.create_dataset(key, data=value)
            elif value is None:
                group.create_dataset(key, data=h5py.Empty("f"))
                group[key].attrs['is_none'] = True
            else:
                # For other types, try to convert to string
                group.create_dataset(key, data=str(value))
                group[key].attrs['is_string_repr'] = True
        
        for key, value in data.items():
            _save_recursive(f, key, value)


def _load_from_hdf5(filepath):
    """Load data from HDF5 file with proper type reconstruction."""
    def _load_recursive(group):
        result = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                if 'is_tensor_list' in item.attrs:
                    # Reconstruct list of tensors
                    tensor_list = []
                    for i in range(len(item.keys())):
                        tensor_data = item[f'item_{i}'][()]
                        tensor_list.append(_numpy_to_tensor(tensor_data))
                    result[key] = tensor_list
                else:
                    result[key] = _load_recursive(item)
            else:
                data = item[()]
                if 'is_none' in item.attrs:
                    result[key] = None
                elif 'is_tensor' in item.attrs:
                    result[key] = _numpy_to_tensor(data)
                elif 'is_string_list' in item.attrs:
                    result[key] = [s.decode('utf-8') if isinstance(s, bytes) else s for s in data]
                elif 'is_string_repr' in item.attrs:
                    result[key] = data.decode('utf-8') if isinstance(data, bytes) else str(data)
                else:
                    # Handle string decoding for byte strings
                    if isinstance(data, bytes):
                        result[key] = data.decode('utf-8')
                    elif isinstance(data, np.ndarray) and data.dtype.kind in ['U', 'S']:
                        # Handle string arrays
                        if data.shape == ():
                            result[key] = str(data)
                        else:
                            result[key] = [str(item) for item in data]
                    else:
                        result[key] = data
        return result
    
    with h5py.File(filepath, 'r') as f:
        return _load_recursive(f)


def save_async(out_dir, file_name, data, file_type):
    """Save data to HDF5 file asynchronously."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    filepath = os.path.join(out_dir, f"{os.path.basename(file_name)}_{file_type}.h5")
    _save_to_hdf5(data, filepath)


def async_load(out_dir, file_name, file_type):
    """Load data from HDF5 file asynchronously."""
    filepath = os.path.join(out_dir, f"{os.path.basename(file_name)}_{file_type}.h5")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} does not exist")

    return _load_from_hdf5(filepath)


def save_latents_hdf5(output_path, file_name, latents, codes):
    """Save latent vectors and codes to HDF5 file."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    filepath = os.path.join(output_path, f"{os.path.basename(file_name)}_latents.h5")
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('latents', data=_tensor_to_numpy(latents))
        f.create_dataset('codes', data=_tensor_to_numpy(codes))
        f['latents'].attrs['is_tensor'] = True
        f['codes'].attrs['is_tensor'] = True


def load_latents_hdf5(latents_path):
    """Load latent vectors and codes from HDF5 file."""
    if not os.path.exists(latents_path):
        raise FileNotFoundError(f"{latents_path} does not exist")
    
    with h5py.File(latents_path, 'r') as f:
        latents = _numpy_to_tensor(f['latents'][()])
        codes = _numpy_to_tensor(f['codes'][()])
    
    return {"latents": latents, "codes": codes}


def represent_encoding(
    midi,
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
