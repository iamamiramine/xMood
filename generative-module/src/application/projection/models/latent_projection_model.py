import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

from application.encoder.models.vocab_model import RemiVocab
from domain.constants.encoder.token_constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN


class StructuralLosses(nn.Module):
    def __init__(self, continuity_weight=0.1, periodicity_weight=0.1):
        super().__init__()
        self.continuity_weight = continuity_weight
        self.periodicity_weight = periodicity_weight
        
    def continuity_loss(self, latent_sequence):
        """Enforce smoothness between consecutive latent vectors"""
        # Calculate differences between consecutive vectors
        diffs = latent_sequence[:, 1:] - latent_sequence[:, :-1]
        # Minimize the magnitude of differences
        return torch.mean(torch.norm(diffs, dim=-1))
    
    def periodicity_loss(self, latent_sequence, bar_length=4):
        """Encourage periodic patterns in the latent sequence"""
        # Compare vectors that should be at similar positions within bars
        periodic_diffs = latent_sequence[:, bar_length:] - latent_sequence[:, :-bar_length]
        # Encourage similarity between corresponding positions
        return torch.mean(torch.norm(periodic_diffs, dim=-1))
    
    def forward(self, latent_sequence):
        c_loss = self.continuity_loss(latent_sequence)
        p_loss = self.periodicity_loss(latent_sequence)
        
        return {
            'continuity_loss': self.continuity_weight * c_loss,
            'periodicity_loss': self.periodicity_weight * p_loss
        }


class LocalAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(d_model, num_heads=num_heads)
        
    def forward(self, x, key_padding_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Create local attention mask
        local_mask = torch.ones(seq_len, seq_len, device=x.device) * float('-inf')
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + 1)  # +1 for causal attention
            local_mask[i, start:end] = 0
            
        # Transpose for attention
        x = x.transpose(0, 1)
        attn_output, _ = self.attention(x, x, x, attn_mask=local_mask, key_padding_mask=key_padding_mask)
        return attn_output.transpose(0, 1)


class LatentProjectionModel(pl.LightningModule):
    def __init__(
        self,
        d_model=512,
        d_latent=512,
        context_size=256,
        num_composers=100,  # Adjust based on your dataset
        num_genres=20,      # Adjust based on your dataset
        window_size=8,
        num_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_positions=512,
        lr=1e-4,
        lr_schedule="sqrt_decay",
        warmup_steps=1000,
        max_steps=10000,
        continuity_weight=0.1,
        periodicity_weight=0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_latent = d_latent
        self.context_size = context_size
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Embeddings for global conditions
        self.composer_embedding = nn.Embedding(num_composers, d_model)
        self.genre_embedding = nn.Embedding(num_genres, d_model)
        
        # Transformer configs
        encoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_positions,
            position_embedding_type="relative_key_query",
        )
        decoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_positions,
            position_embedding_type="relative_key_query",
        )
        
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.transformer = EncoderDecoderModel(config)
        self.transformer.config.decoder.is_decoder = True
        self.transformer.config.decoder.add_cross_attention = True
        
        # Local attention layers
        self.local_attention = LocalAttention(d_model, num_attention_heads, window_size)
        
        # Output projection to latent space
        self.to_latent = nn.Linear(d_model, d_latent)
        
        # Structural losses
        self.structural_losses = StructuralLosses(continuity_weight, periodicity_weight)
        
        # MSE loss for latent reconstruction
        self.mse_loss = nn.MSELoss()
        
        self.save_hyperparameters()

    def encode_conditions(self, composer_ids, genre_ids):
        """Encode composer and genre conditions"""
        composer_emb = self.composer_embedding(composer_ids)
        genre_emb = self.genre_embedding(genre_ids)
        
        # Combine embeddings
        combined_emb = composer_emb + genre_emb
        
        # Pass through encoder
        encoder_outputs = self.transformer.encoder(inputs_embeds=combined_emb)
        return encoder_outputs.last_hidden_state

    def decode(self, encoder_hidden_states, target_length):
        """Decode latent sequence from encoded conditions"""
        batch_size = encoder_hidden_states.size(0)
        
        # Initialize sequence with zeros
        sequence = torch.zeros(
            batch_size, target_length, self.d_model,
            device=self.device
        )
        
        # Generate sequence autoregressively
        for i in range(target_length):
            # Apply local attention to current sequence
            local_features = self.local_attention(sequence[:, :i+1])
            
            # Cross attention with encoded conditions
            decoder_outputs = self.transformer.decoder(
                inputs_embeds=local_features,
                encoder_hidden_states=encoder_hidden_states
            )
            
            # Update sequence
            sequence[:, i] = decoder_outputs.last_hidden_state[:, -1]
        
        # Project to latent space
        return self.to_latent(sequence)

    def forward(self, composer_ids, genre_ids, target_length):
        # Encode conditions
        encoder_hidden = self.encode_conditions(composer_ids, genre_ids)
        
        # Generate latent sequence
        latents = self.decode(encoder_hidden, target_length)
        
        # Calculate structural losses during training
        if self.training:
            losses = self.structural_losses(latents)
            return latents, losses
            
        return latents

    def training_step(self, batch, batch_idx):
        composer_ids = batch['composer_ids']
        genre_ids = batch['genre_ids']
        target_latents = batch['latents']
        
        # Generate latent sequence
        pred_latents, structural_losses = self(
            composer_ids,
            genre_ids,
            target_latents.size(1)
        )
        
        # Calculate losses
        reconstruction_loss = self.mse_loss(pred_latents, target_latents)
        total_loss = (
            reconstruction_loss +
            structural_losses['continuity_loss'] +
            structural_losses['periodicity_loss']
        )
        
        # Log losses
        self.log('train/reconstruction_loss', reconstruction_loss)
        self.log('train/continuity_loss', structural_losses['continuity_loss'])
        self.log('train/periodicity_loss', structural_losses['periodicity_loss'])
        self.log('train/total_loss', total_loss)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        composer_ids = batch['composer_ids']
        genre_ids = batch['genre_ids']
        target_latents = batch['latents']
        
        # Generate latent sequence
        pred_latents = self(
            composer_ids,
            genre_ids,
            target_latents.size(1)
        )
        
        # Calculate validation loss
        val_loss = self.mse_loss(pred_latents, target_latents)
        self.log('val/loss', val_loss)
        
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1, weight_decay=0.01)
        
        if self.lr_schedule == "sqrt_decay":
            lr_func = lambda step: min(
                self.lr,
                self.lr / math.sqrt(max(step, 1) / self.warmup_steps)
            )
        elif self.lr_schedule == "linear":
            lr_func = lambda step: min(
                self.lr,
                self.lr * step / self.warmup_steps,
                self.lr * (1 - (step - self.warmup_steps) / self.max_steps)
            )
        elif self.lr_schedule == "cosine":
            lr_func = lambda step: self.lr * min(
                step / self.warmup_steps,
                0.55 + 0.45 * math.cos(math.pi * (min(step, self.max_steps) - self.warmup_steps) / (self.max_steps - self.warmup_steps))
            )
        else:
            lr_func = lambda step: self.lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @torch.no_grad()
    def generate(self, composer_ids, genre_ids, sequence_length, temperature=1.0):
        """Generate a latent sequence from composer and genre conditions"""
        # Generate base sequence
        latents = self(composer_ids, genre_ids, sequence_length)
        
        # Apply temperature
        if temperature != 1.0:
            latents = latents / temperature
            
        return latents 