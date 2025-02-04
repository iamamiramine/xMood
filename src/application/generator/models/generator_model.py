import lightning.pytorch as pl
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math

from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

from application.encoder.models.vocab_model import RemiVocab, SymbolicFeaturesVocab
from domain.constants.encoder.token_constants import PAD_TOKEN, EOS_TOKEN, BAR_KEY, POSITION_KEY, BOS_TOKEN


class GroupEmbedding(nn.Module):
    def __init__(self, n_tokens, n_groups, out_dim, inner_dim=128):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_groups = n_groups
        self.inner_dim = inner_dim
        self.out_dim = out_dim

        self.embedding = nn.Embedding(n_tokens, inner_dim)
        self.proj = nn.Linear(n_groups * inner_dim, out_dim, bias=False)

    def forward(self, x):
        shape = x.shape
        emb = self.embedding(x)
        return self.proj(emb.view(*shape[:-1], self.n_groups * self.inner_dim))


def get_embedding(tensor, embedding_fn, device):
    if tensor is not None:
        return embedding_fn(tensor.to(device))
    return None


def pad_and_transpose(embeddings):
    transposed = [emb.transpose(0, 1) for emb in embeddings if emb is not None]
    padded = pad_sequence(transposed, batch_first=True)
    return padded.transpose(1, 2)


def concatenate_embeddings(*embeddings):
    return torch.cat([emb for emb in embeddings if emb is not None], dim=-1)


class MIDIGeneratorModule(pl.LightningModule):
    def __init__(
        self,
        d_model=512,
        d_latent=512,
        n_codes=512,
        n_groups=8,
        context_size=-1,
        max_bars=512,
        max_positions=512,
        lr=1e-4,
        lr_schedule="sqrt_decay",
        warmup_steps=None,
        max_steps=None,
        encoder_layers=6,
        decoder_layers=12,
        intermediate_size=2048,
        num_attention_heads=8,
        use_pretrained_latent_embeddings=True,
        device="cuda:0",
    ):
        super(MIDIGeneratorModule, self).__init__()

        self.context_size = context_size
        self.max_bars = max_bars
        self.max_positions = max_positions
        self.d_model = d_model
        self.d_latent = d_latent

        self.lr = lr
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        self._device = torch.device(device)

        self.vocab = RemiVocab()
        self.symb_vocab = SymbolicFeaturesVocab()

        encoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=self.d_model,
            num_hidden_layers=encoder_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=self.max_positions,
            position_embedding_type="relative_key_query",
        )
        decoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=self.d_model,
            num_hidden_layers=decoder_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=self.max_positions,
            position_embedding_type="relative_key_query",
        )
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

        self.transformer = EncoderDecoderModel(config)

        self.transformer.config.decoder.is_decoder = True
        self.transformer.config.decoder.add_cross_attention = True

        self.bar_embedding = nn.Embedding(self.max_bars + 1, self.d_model)
        self.pos_embedding = nn.Embedding(self.max_positions + 1, self.d_model)

        if use_pretrained_latent_embeddings:
            self.latent_in = nn.Linear(self.d_latent, self.d_model, bias=False)
        else:
            self.latent_in = GroupEmbedding(n_codes, n_groups, self.d_model, inner_dim=self.d_latent // n_groups)

        self.symb_in = nn.Embedding(len(self.symb_vocab), self.d_model)

        self.symb_proj = nn.Linear(2 * self.d_model, self.d_model, bias=False)

        self.in_layer = nn.Embedding(len(self.vocab), self.d_model)
        self.out_layer = nn.Linear(self.d_model, len(self.vocab), bias=False)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.to_i(PAD_TOKEN))

        self.save_hyperparameters()

    def encode(self, z=None, symb_bar_ids=None):
        latent_emb, symb_emb = None, None
        if "bar_symbolic_ids" in z:
            symb_emb = get_embedding(z.get("bar_symbolic_ids"), self.symb_in, self._device)

        if "latents" in z:
            latent_emb = get_embedding(z.get("latents"), self.latent_in, self._device)

        # Collect non-None embeddings
        embeddings = [emb for emb in [symb_emb, latent_emb] if emb is not None]

        # Handle case when z is None
        if not embeddings:
            return None

        # Pad and transpose embeddings
        padded_embeddings = pad_and_transpose(embeddings)

        if len(padded_embeddings) == 2:
            symb_emb, latent_emb = padded_embeddings
        elif len(padded_embeddings) == 1:
            if symb_emb is not None:
                symb_emb = padded_embeddings[0]
            elif latent_emb is not None:
                latent_emb = padded_embeddings[0]

        # Collect non-None embeddings
        z_embeddings = [emb for emb in [symb_emb, latent_emb] if emb is not None]

        # Determine the output
        if len(z_embeddings) == 1:
            z_emb = embeddings[0]  # Return the single embedding as is
            if symb_bar_ids is not None and symb_emb is not None:
                z_emb += self.bar_embedding(symb_bar_ids.to(self._device))
        elif len(z_embeddings) > 1:
            if symb_bar_ids is not None and symb_emb is not None:
                symb_emb = symb_emb + self.bar_embedding(symb_bar_ids.to(self._device))
                z_embeddings = [emb for emb in [symb_emb, latent_emb] if emb is not None]
            z_emb = self.symb_proj(concatenate_embeddings(*z_embeddings))  # Project concatenated embeddings

        out = self.transformer.encoder(inputs_embeds=z_emb, output_hidden_states=True)
        encoder_hidden = out.hidden_states[-1]
        return encoder_hidden

    def decode(self, x, labels=None, bar_ids=None, position_ids=None, encoder_hidden_states=None, return_hidden=False):
        seq_len = x.size(1)

        # Shape of x_emb: (batch_size, seq_len, d_model)
        x_emb = self.in_layer(x)
        if bar_ids is not None:
            bar_ids = torch.clamp(bar_ids, max=self.max_bars)
            x_emb += self.bar_embedding(bar_ids)
        if position_ids is not None:
            position_ids = torch.clamp(position_ids, max=self.max_positions)
            x_emb += self.pos_embedding(position_ids)

        if encoder_hidden_states is not None:
            # Make x_emb and encoder_hidden_states match in sequence length. Necessary for relative positional embeddings
            padded = pad_sequence([x_emb.transpose(0, 1), encoder_hidden_states.transpose(0, 1)], batch_first=True)
            x_emb, encoder_hidden_states = padded.transpose(1, 2)

            out = self.transformer.decoder(inputs_embeds=x_emb, encoder_hidden_states=encoder_hidden_states, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, :seq_len]
        else:
            out = self.transformer.decoder(inputs_embeds=x_emb, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, :seq_len]

        # Shape of logits: (batch_size, seq_len, tuple_size, vocab_size)
        if return_hidden:
            return hidden
        else:
            return self.out_layer(hidden)

    def forward(self, x, z=None, labels=None, position_ids=None, bar_ids=None, symbolic_bar_ids=None, return_hidden=False):
        encoder_hidden = self.encode(z, symb_bar_ids=symbolic_bar_ids)

        out = self.decode(
            x,
            labels=labels,
            bar_ids=bar_ids,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden,
            return_hidden=return_hidden,
        )

        return out

    def _build_z_input(self, batch):
        """
        Builds the z input dictionary from batch data.
        Returns a tuple of (z, symb_bar_ids) where:
        - z: dictionary containing latents, symbolic
        - symb_bar_ids: symbolic bar ids if symbolic data is present, else None
        """
        if batch is None:
            return None, None

        available_inputs = {
            "latents": batch.get("latents"),
            "bar_symbolic": batch.get("bar_symbolic_ids"),
        }

        # Filter out None values
        z = {k: v for k, v in available_inputs.items() if v is not None}

        # If no inputs are available, return None
        if not z:
            return None, None

        symb_bar_ids = batch.get("symb_bar_ids") if "symbolic" in z else None
        return z, symb_bar_ids

    def get_loss(self, batch, return_logits=False):
        x = batch["input_ids"]  # Shape of x: (batch_size, seq_len, tuple_size)
        bar_ids = batch["bar_ids"]
        position_ids = batch["position_ids"]
        labels = batch["labels"]  # Shape of labels: (batch_size, tgt_len, tuple_size)

        z, symb_bar_ids = self._build_z_input(batch)
        logits = self(x, z=z, labels=labels, bar_ids=bar_ids, position_ids=position_ids, symbolic_bar_ids=symb_bar_ids)
        # Shape of logits: (batch_size, tgt_len, tuple_size, vocab_size)
        pred = logits.view(-1, logits.shape[-1])
        labels = labels.reshape(-1)

        loss = self.loss_fn(pred, labels)

        if return_logits:
            return loss, logits
        else:
            return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.get_loss(batch, return_logits=True)
        self.log("valid_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        y = batch["labels"]
        pad_token_id = self.vocab.to_i(PAD_TOKEN)

        logits = logits.view(logits.size(0), -1, logits.size(-1))
        y = y.view(y.size(0), -1)

        log_pr = logits.log_softmax(dim=-1)
        log_pr[y == pad_token_id] = 0  # log(pr) = log(1) for padding
        log_pr = torch.gather(log_pr, -1, y.unsqueeze(-1)).squeeze(-1)

        t = (y != pad_token_id).sum(dim=-1)
        ppl = (-log_pr.sum(dim=1) / t).exp().mean()
        self.log("valid_ppl", ppl.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.get_loss(batch)

    def configure_optimizers(self):
        # set LR to 1, scale with LambdaLR scheduler
        optimizer = torch.optim.AdamW(self.parameters(), lr=1, weight_decay=0.01)

        if self.lr_schedule == "sqrt_decay":
            # constant warmup, then 1/sqrt(n) decay starting from the initial LR
            lr_func = lambda step: min(self.lr, self.lr / math.sqrt(max(step, 1) / self.warmup_steps))
        elif self.lr_schedule == "linear":
            # linear warmup, linear decay
            lr_func = lambda step: min(self.lr, self.lr * step / self.warmup_steps, self.lr * (1 - (step - self.warmup_steps) / self.max_steps))
        elif self.lr_schedule == "cosine":
            # linear warmup, cosine decay to 10% of initial LR
            lr_func = lambda step: self.lr * min(
                step / self.warmup_steps,
                0.55 + 0.45 * math.cos(math.pi * (min(step, self.max_steps) - self.warmup_steps) / (self.max_steps - self.warmup_steps)),
            )
        else:
            # Use no lr scheduling
            lr_func = lambda step: self.lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
            }
        ]

    @torch.no_grad()
    def sample(self, batch=None, max_length=256, max_bars=-1, temp=0.8, pad_token=PAD_TOKEN, eos_token=EOS_TOKEN, verbose=0):
        # Setup and parsing arguments
        pad_token_id = torch.tensor(self.vocab.to_i(pad_token)).to(self._device)
        eos_token_id = torch.tensor(self.vocab.to_i(eos_token)).to(self._device)

        x = torch.tensor(self.vocab.encode([BOS_TOKEN]), dtype=torch.int).unsqueeze(0).to(self._device)
        position_ids = torch.tensor([0], dtype=torch.int).unsqueeze(0).to(self._device)
        bar_ids = torch.tensor([0], dtype=torch.int).unsqueeze(0).to(self._device)

        assert (
            x.shape[:2] == bar_ids.shape and x.shape[:2] == position_ids.shape
        ), f"Input, bar and position ids weren't of compatible shapes: {x.shape}, {bar_ids.shape}, {position_ids.shape}"

        batch_size, curr_len = x.shape
        i = curr_len - 1

        # Get z input using the helper function
        z, symb_bar_ids = self._build_z_input(batch)

        is_done = torch.zeros(batch_size, dtype=torch.bool).to(self._device)

        # Precompute encoder hidden states for cross-attention
        encoder_hidden_states = None
        if z is not None and "latents" in z:  # TODO: Check if we include symbolic features and emotions in pre-compute hidden states
            encoder_hidden_states = self.encode(z, symb_bar_ids)

        curr_bars = torch.zeros(batch_size).fill_(-1).to(self._device)
        # Sample using decoder until max_length is reached or all sequences are done
        for i in range(curr_len - 1, max_length):
            x_ = x[:, -self.context_size :].to(self._device)
            bar_ids_ = bar_ids[:, -self.context_size :].to(self._device)
            position_ids_ = position_ids[:, -self.context_size :].to(self._device)

            # symbolic scrolling
            if z is not None:
                if z.get("symbolic") is not None:
                    desc = z["symbolic"]

                    next_bars = bar_ids_[:, 0]
                    bars_changed = not (next_bars == curr_bars).all()
                    curr_bars = next_bars

                    if bars_changed:
                        z_ = torch.zeros(batch_size, self.context_size, dtype=torch.int)
                        symb_bar_ids_ = torch.zeros(batch_size, self.context_size, dtype=torch.int).to(self._device)

                        for j in range(batch_size):
                            curr_bar = bar_ids_[j, 0].to(self._device)
                            indices = torch.nonzero(symb_bar_ids[j].to(self._device) == curr_bar)
                            if indices.size(0) > 0:
                                idx = indices[0, 0]
                            else:
                                idx = desc.size(1) - 1

                            offset = min(self.context_size, desc.size(1) - idx)

                            z_[j, :offset] = desc[j, idx : idx + offset]
                            symb_bar_ids_[j, :offset] = symb_bar_ids[j, idx : idx + offset]

                        z_, symb_bar_ids_ = z_.to(self._device), symb_bar_ids_.to(self._device)
                        z_ = {"symbolic": z_}

                        if z.get("latents") is not None:
                            z_["latents"] = z["latents"].to(self._device)

                        encoder_hidden_states = self.encode(z_, symb_bar_ids_)

            logits = self.decode(x_, bar_ids=bar_ids_, position_ids=position_ids_, encoder_hidden_states=encoder_hidden_states)

            idx = min(self.context_size - 1, i)
            logits = logits[:, idx] / temp

            # # optionally crop the logits to only the top k options
            # if top_k is not None:
            #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            #     logits[logits < v[:, [-1]]] = -float("Inf")

            pr = F.softmax(logits, dim=-1)
            pr = pr.view(-1, pr.size(-1))

            next_token_ids = torch.multinomial(pr, 1).view(-1).to(self._device)
            next_tokens = self.vocab.decode(next_token_ids)
            if verbose:
                print(f"{i+1}/{max_length}", next_tokens, flush=True)

            next_bars = torch.tensor([1 if f"{BAR_KEY}_" in token else 0 for token in next_tokens], dtype=torch.int).to(self._device)
            next_bar_ids = bar_ids[:, i].clone() + next_bars

            next_positions = [f"{POSITION_KEY}_0" if f"{BAR_KEY}_" in token else token for token in next_tokens]
            next_positions = [int(token.split("_")[-1]) if f"{POSITION_KEY}_" in token else None for token in next_positions]
            next_positions = [pos if next_pos is None else next_pos for pos, next_pos in zip(position_ids[:, i], next_positions)]
            next_position_ids = torch.tensor(next_positions, dtype=torch.int).to(self._device)

            is_done.masked_fill_((next_token_ids == eos_token_id).all(dim=-1), True)
            next_token_ids[is_done] = pad_token_id
            if max_bars > 0:
                is_done.masked_fill_(next_bar_ids >= max_bars + 1, True)

            x = torch.cat([x, next_token_ids.clone().unsqueeze(1)], dim=1)
            bar_ids = torch.cat([bar_ids, next_bar_ids.unsqueeze(1)], dim=1)
            position_ids = torch.cat([position_ids, next_position_ids.unsqueeze(1)], dim=1)

            if torch.all(is_done):
                break

        return {"sequences": x, "bar_ids": bar_ids, "position_ids": position_ids}
