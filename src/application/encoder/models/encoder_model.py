import os
import torch
import lightning.pytorch as pl
import pretty_midi as pm

from application.encoder.helpers.remi_helper import get_remi_events
from application.encoder.models.item_model import Item
from application.encoder.helpers.encoder_helper import (
    read_note_tempo,
    quantize_midi,
    extract_beats,
    extract_downbeats,
    group_items,
    extract_dominant_keys,
)

from application.music_base.services.music_base_service import (
    extract_chords,
    estimate_tonal_plan,
)

from domain.models.music_base.music_base_model import (
    TonalPlanParameters,
    MusicBaseParameters,
)

from persistence.dataloader.repositories.dataloader_repository import (
    save_async,
    async_load,
)


class MIDIEncoderModule(pl.LightningModule):
    def __init__(
        self,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        c=1.0,
        w=1.0,
        device="cuda:0",
        save=False,
        encodings_out_dir=None,
    ):
        super(MIDIEncoderModule, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c = c
        self.w = w
        self.save = save
        self.encodings_out_dir = encodings_out_dir
        self._device = torch.device(device)

    def encode_midi(self, midi_path):
        """
        Encode a single MIDI file into a symbolic representation with harmonic analysis.
        """
        # Step 1: Load MIDI file
        midi = pm.PrettyMIDI(midi_path)
        sample = {}

        # Step 2: Extract basic MIDI features
        note_items, tempo_items = read_note_tempo(midi)
        quantize_midi(midi, note_items, midi.resolution)
        beats = extract_beats(midi)
        downbeats = extract_downbeats(midi)

        # Check if processed file already exists
        processed_path = os.path.join(self.encodings_out_dir, f"{os.path.basename(midi_path)}_processed.pkl")
        existing_data = None
        if os.path.isfile(processed_path):
            try:
                existing_data = async_load(self.encodings_out_dir, midi_path, "processed")
            except Exception:
                existing_data = None

        # Step 3: Harmonic Analysis - Chord Extraction
        if existing_data and "chords" in existing_data:
            if "chords" in existing_data["chords"] and "remi_chords" in existing_data["chords"]:
                chords = {"chords": existing_data["chords"]["chords"]}
                remi_chords = existing_data["chords"]["remi_chords"]
            elif "chords" in existing_data["chords"]:
                chords = {"chords": existing_data["chords"]["chords"]}
                remi_chords = []
                for chord in chords["chords"]:
                    remi_chords.append(
                        Item(
                            name="Chord",
                            start=midi.time_to_tick(chord[0]),
                            end=midi.time_to_tick(chord[1]),
                            velocity=None,
                            pitch=chord[2].split("/")[0],
                        )
                    )
                if len(remi_chords) == 0 or remi_chords[0].start > 0:
                    end = midi.time_to_tick(midi.get_end_time()) if len(remi_chords) == 0 else remi_chords[0].start
                    remi_chords.append(Item(name="Chord", start=0, end=end, velocity=None, pitch="N:N"))
            else:
                remi_chords = existing_data["chords"]["remi_chords"]
                chords = {"chords": []}
        else:
            chords = extract_chords(
                MusicBaseParameters(
                    midi=midi_path,
                    save=False,
                ),
            )
            remi_chords = []
            for chord in chords["chords"]:
                remi_chords.append(
                    Item(
                        name="Chord",
                        start=midi.time_to_tick(chord[0]),
                        end=midi.time_to_tick(chord[1]),
                        velocity=None,
                        pitch=chord[2].split("/")[0],
                    )
                )
            if len(remi_chords) == 0 or remi_chords[0].start > 0:
                end = midi.time_to_tick(midi.get_end_time()) if len(remi_chords) == 0 else remi_chords[0].start
                remi_chords.append(Item(name="Chord", start=0, end=end, velocity=None, pitch="N:N"))

        # Step 4: Harmonic Analysis - Key Detection
        if existing_data and "keys" in existing_data:
            if "keys" in existing_data["keys"] and "remi_keys" in existing_data["keys"]:
                keys = {"keys": existing_data["keys"]["keys"]}
                remi_keys = existing_data["keys"]["remi_keys"]
            elif "keys" in existing_data["keys"]:
                keys = {"keys": existing_data["keys"]["keys"]}
                remi_keys = []
                for i in range(1, len(beats), 1):
                    remi_keys.append(
                        Item(
                            name="Key",
                            start=midi.time_to_tick(beats[i - 1]),
                            end=midi.time_to_tick(beats[i]),
                            velocity=None,
                            pitch=f"{keys['keys'][i-1][0]}:{keys['keys'][i-1][1]}",
                        )
                    )
            else:
                remi_keys = existing_data["keys"]["remi_keys"]
                keys = {"keys": []}
        else:
            keys = estimate_tonal_plan(
                TonalPlanParameters(
                    midi=midi_path,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                    c=self.c,
                    w=self.w,
                    save=False,
                    chords=chords,
                ),
            )
            remi_keys = []
            for i in range(1, len(beats), 1):
                remi_keys.append(
                    Item(
                        name="Key",
                        start=midi.time_to_tick(beats[i - 1]),
                        end=midi.time_to_tick(beats[i]),
                        velocity=None,
                        pitch=f"{keys['keys'][i-1][0]}:{keys['keys'][i-1][1]}",
                    )
                )

        # Step 5: Combine all musical events
        midi.tonal_plan = remi_keys
        items = remi_keys + remi_chords + tempo_items + note_items
        groups = group_items(midi, downbeats, items=items)
        groups = extract_dominant_keys(groups)
        sample["events"] = get_remi_events(midi, groups)[1]

        # Step 6: Combine all outputs
        processed_output = {
            "encoding": sample,
            "chords": {"chords": chords["chords"], "remi_chords": remi_chords},
            "keys": {"keys": keys["keys"], "remi_keys": remi_keys},
        }

        # Step 7: Save output if requested
        if self.save:
            save_async(self.encodings_out_dir, midi_path, processed_output, "processed")

        # demo = async_load(self.encodings_out_dir, midi_path, "processed")
        #
        # print(demo, flush=True)

        return processed_output

    def training_step(self, batch, batch_idx):
        # This is a placeholder since we're not actually training the encoder
        return None

    def validation_step(self, batch, batch_idx):
        # This is a placeholder since we're not actually training the encoder
        return None

    def predict_step(self, batch, batch_idx):
        """Process a batch of MIDI files for encoding."""
        results = []
        for file_path in batch["file"]:
            try:
                result = self.encode_midi(file_path)
                results.append({"success": True, "file": file_path})
            except Exception as e:
                results.append({"success": False, "file": file_path, "error": str(e)})
        return results

    def configure_optimizers(self):
        # This is a placeholder since we're not actually training the encoder
        return None
