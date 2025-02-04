import lightning.pytorch as pl
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math

from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

from application.encoder.models.vocab_model import RemiVocab, SymbolicFeaturesVocab, EmotionVocab
from domain.constants.encoder.token_constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, BAR_KEY, POSITION_KEY


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


class ProjectorModule(pl.LightningModule):
    def __init__(
        self,
        d_model=512,
        d_latent=512,
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
        device="cuda:0",
    ):
        super(ProjectorModule, self).__init__()

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
        self.emotion_vocab = EmotionVocab()

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

        # Input layers for different feature types
        self.piece_symb_in = nn.Embedding(len(self.symb_vocab), self.d_model)
        self.emotions_in = nn.Embedding(len(self.emotion_vocab), self.d_model)
        self.prompt_proj = nn.Linear(2 * self.d_model, self.d_model, bias=False)

        self.bar_embedding = nn.Embedding(self.max_bars + 1, self.d_model)
        self.pos_embedding = nn.Embedding(self.max_positions + 1, self.d_model)

        self.in_layer = nn.Embedding(len(self.symb_vocab), self.d_model)
        self.input_proj = nn.Linear(2 * self.d_model, self.d_model)
        self.out_layer = nn.Linear(self.d_model, len(self.symb_vocab), bias=False)

        # Loss functions
        self.symbolic_loss = nn.CrossEntropyLoss(ignore_index=self.vocab.to_i(PAD_TOKEN))

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.to_i(PAD_TOKEN))

        self.save_hyperparameters()

    def encode(self, z=None):
        piece_symb_emb, piece_emotions_emb = None, None
        if "piece_symbolic_ids" in z:
            piece_symb_emb = get_embedding(z.get("piece_symbolic_ids"), self.piece_symb_in, self._device)

        if "piece_emotions_ids" in z:
            piece_emotions_emb = get_embedding(z.get("piece_emotions_ids"), self.emotions_in, self._device)

        # Collect non-None embeddings
        embeddings = [emb for emb in [piece_symb_emb, piece_emotions_emb] if emb is not None]

        # Handle case when z is None
        if not embeddings:
            return None

        # Pad and transpose embeddings
        padded_embeddings = pad_and_transpose(embeddings)

        if len(padded_embeddings) == 2:
            piece_symb_emb, piece_emotions_emb = padded_embeddings
        elif len(padded_embeddings) == 1:
            if piece_symb_emb is not None:
                piece_symb_emb = padded_embeddings[0]
            elif piece_emotions_emb is not None:
                piece_emotions_emb = padded_embeddings[0]

        # Collect non-None embeddings
        z_embeddings = [emb for emb in [piece_symb_emb, piece_emotions_emb] if emb is not None]

        # Determine the output
        if len(z_embeddings) == 1:
            z_emb = z_embeddings[0]  # Return the single embedding as is
        elif len(z_embeddings) > 1:
            z_emb = self.prompt_proj(concatenate_embeddings(*z_embeddings))  # Project concatenated embeddings

        out = self.transformer.encoder(inputs_embeds=z_emb, output_hidden_states=True)
        encoder_hidden = out.hidden_states[-1]
        return encoder_hidden

    def decode(self, x, encoder_hidden_states=None, bar_ids=None, position_ids=None):
        """Decode encoder hidden states into different feature types.
        Args:
            x: piece-level symbolic features [batch_size, n_features]
            encoder_hidden_states: Encoded emotion features [batch_size, seq_len, d_model]
            bar_ids: Bar position indices [batch_size, target_seq_len] (optional, provided during training)
            position_ids: Position indices [batch_size, target_seq_len] (optional, provided during training)
        Returns:
            Logits for bar-level symbolic features
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        x_emb = self.in_layer(x)  # [batch_size, seq_len, d_model]
        if bar_ids is not None:
            bar_ids = torch.clamp(bar_ids, max=self.max_bars)
            x_emb = self.bar_embedding(bar_ids)
        if position_ids is not None:
            position_ids = torch.clamp(position_ids, max=self.max_positions)
            x_emb += self.pos_embedding(position_ids)

        if encoder_hidden_states is not None:
            padded = pad_sequence([x_emb.transpose(0, 1), encoder_hidden_states.transpose(0, 1)], batch_first=True)
            x_emb, encoder_hidden_states = padded.transpose(1, 2)

            out = self.transformer.decoder(inputs_embeds=x_emb, encoder_hidden_states=encoder_hidden_states, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, :seq_len]
        else:
            out = self.transformer.decoder(inputs_embeds=x_emb, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, :seq_len]

        return self.out_layer(hidden)

    def forward(self, x, z, bar_ids=None, position_ids=None):
        """
        Forward pass of the model.
        Args:
            x: bar_symbolic_ids for teacher forcing during training
            z: Dictionary containing input features (emotions, symbolic)
            bar_ids: Bar position indices [batch_size, seq_len]
        Returns:
            Dictionary containing generated features
        """
        # Validate inputs
        if z is None:
            raise ValueError("No conditioning tags provided (emotions or symbolic features)")

        encoder_hidden = self.encode(z)
        features = self.decode(x, encoder_hidden, bar_ids, position_ids)
        return features

    def _build_z_input(self, batch):
        """
        Builds the z input dictionary from batch data.
        Args:
            batch: Dictionary containing input features
        Returns:
            Dictionary containing the input features or None if no valid inputs
        """
        if batch is None:
            return None

        available_inputs = {
            "piece_emotions_ids": batch.get("piece_emotions_ids"),
            "piece_symbolic_ids": batch.get("piece_symbolic_ids"),
        }

        # Filter out None values
        z = {k: v for k, v in available_inputs.items() if v is not None}

        # If no inputs are available, return None
        if not z:
            return None

        return z

    def get_loss(self, batch, return_logits=False):
        """
        Calculate loss for the batch.
        Args:
            batch: Dictionary containing input and target features
            return_logits: Whether to return model outputs along with loss
        Returns:
            total_loss if return_logits is False, else (total_loss, logits)
        """
        # Get input sequence (all tokens except last) and target sequence (all tokens except first)
        x = batch["bar_symbolic_ids"][:, :-1]  # Input sequence
        labels = batch["bar_symbolic_ids"][:, 1:].clone().detach()  # Target sequence

        bar_ids = None
        if "symb_bar_ids" in batch:
            bar_ids = batch["symb_bar_ids"]
        position_ids = None
        if "symb_position_ids" in batch:
            position_ids = batch["symb_position_ids"]

        # Build z input with piece-level features
        z = self._build_z_input(batch)

        logits = self(x, z, bar_ids, position_ids)

        pred = logits.view(-1, logits.shape[-1])
        labels = labels.reshape(-1).long()  # Ensure labels are of type Long
        # Standard cross entropy loss
        loss = self.loss_fn(pred, labels)

        self.log("train_ce_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss if not return_logits else (loss, logits)

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        loss = self.get_loss(batch)
        print(loss, flush=True)
        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        loss, logits = self.get_loss(batch, return_logits=True)
        self.log("valid_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        y = batch["bar_symbolic_ids"]
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
        """Test step for the model."""
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
    def sample(self, batch=None, max_length=256, temp=0.8, pad_token=PAD_TOKEN, eos_token=EOS_TOKEN, verbose=0):
        """
        Generate a sequence of bar-level symbolic features using input tags.
        Args:
            batch: Dictionary containing input features
            max_length: Maximum sequence length to generate
            temp: Temperature for sampling (higher = more random)
            pad_token: Token to use for padding
            eos_token: Token to use for end of sequence
            verbose: Whether to print generation progress
        Returns:
            Dictionary containing generated sequences
        """
        # Convert tokens to ids
        pad_token_id = self.symb_vocab.to_i(pad_token)
        eos_token_id = self.symb_vocab.to_i(eos_token)

        # Initialize sequence with start token
        x = torch.tensor(self.symb_vocab.encode([BOS_TOKEN]), dtype=torch.int).unsqueeze(0).to(self._device)  # previously was [[BOS_TOKEN]]
        position_ids = torch.tensor([0], dtype=torch.int).unsqueeze(0).to(self._device)
        bar_ids = torch.tensor([0], dtype=torch.int).unsqueeze(0).to(self._device)  # previously was [[0]]

        # Get batch size from input
        batch_size, curr_len = x.shape

        # Track which sequences are done
        is_done = torch.zeros(batch_size, dtype=torch.bool).to(self._device)

        # Build z input with piece-level features
        z = self._build_z_input(batch)

        # Precompute encoder hidden states for cross-attention
        encoder_hidden = self.encode(z)

        # Generate sequence
        for i in range(curr_len - 1, max_length):  # previously was range(max_length)
            # Get model predictions
            logits = self.decode(x, encoder_hidden, bar_ids, position_ids)
            
            # Get predictions for next token and apply temperature
            logits = logits[:, -1] / temp

            # Sample from the probability distribution
            pr = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(pr, num_samples=1).view(-1)

            if verbose:
                tokens = [self.symb_vocab.decode([t.item()])[0] for t in next_token]
                print(f"{i+1}/{max_length}", tokens, flush=True)

            # Update bar_ids based on whether the next token is a bar token
            next_tokens = self.symb_vocab.decode(next_token)
            next_bars = torch.tensor([1 if f"{BAR_KEY}_" in token else 0 for token in next_tokens], dtype=torch.int).to(self._device)
            next_bar_ids = bar_ids[:, -1].clone() + next_bars

            # Update position embeddings
            next_positions = [f"{POSITION_KEY}_0" if f"{BAR_KEY}_" in token else token for token in next_tokens]
            next_positions = [int(token.split("_")[-1]) if f"{POSITION_KEY}_" in token else None for token in next_positions]
            next_positions = [pos if next_pos is None else next_pos for pos, next_pos in zip(position_ids[:, i], next_positions)]
            next_position_ids = torch.tensor(next_positions, dtype=torch.int).to(self._device)

            # Update done mask for completed sequences
            is_done = is_done | (next_token == eos_token_id)
            next_token[is_done] = pad_token_id

            # Append new tokens to sequence
            x = torch.cat([x, next_token.unsqueeze(1)], dim=1)
            bar_ids = torch.cat([bar_ids, next_bar_ids.unsqueeze(1)], dim=1)
            position_ids = torch.cat([position_ids, next_position_ids.unsqueeze(1)], dim=1)

            # Early stopping if all sequences are done
            if is_done.all():
                break

        return {"sequences": x, "bar_ids": bar_ids}

    @staticmethod
    def post_process_sequence(sequence):
        """
        Post-process the generated sequence to ensure it follows the correct format.
        Args:
            sequence: List of tokens
        Returns:
            List of tokens after post-processing
        """
        # Remove <bos> token if present
        if sequence[0] == '<bos>':
            sequence = sequence[1:]

        # Find the first occurrence of Bar_1 and remove everything before it
        try:
            bar_1_index = sequence.index('Bar_1')
            sequence = sequence[bar_1_index:]
        except ValueError:
            pass  # If Bar_1 is not found, keep the sequence as is

        # Remove <unk> tokens
        sequence = [token for token in sequence if token != '<unk>']

        # Process the sequence bar by bar
        processed_sequence = []
        current_bar = []
        current_position = None
        allowed_multiple = {'Instrument_', 'Chord_'}

        for token in sequence:
            # Start of a new bar
            if token.startswith('Bar_'):
                if current_bar:
                    processed_sequence.extend(current_bar)
                current_bar = [token]
                current_position = None
            # Position token
            elif token.startswith('Position_'):
                if current_position is not None:
                    processed_sequence.extend(current_bar)
                    current_bar = []
                current_bar.append(token)
                current_position = token
            # Other tokens
            else:
                # Check if token is allowed to have multiple values
                is_allowed_multiple = any(token.startswith(prefix) for prefix in allowed_multiple)
                
                # For tokens that should be unique per position
                if not is_allowed_multiple:
                    # Check if we already have a token of the same type
                    token_type = token.split('_')[0]
                    existing_tokens = [t for t in current_bar if t.split('_')[0] == token_type]
                    if existing_tokens:
                        continue
                
                current_bar.append(token)

        # Add the last bar
        if current_bar:
            processed_sequence.extend(current_bar)

        return processed_sequence
