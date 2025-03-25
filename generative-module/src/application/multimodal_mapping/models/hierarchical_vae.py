import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from application.encoder.helpers.vocab_helper import get_bars
from application.encoder.models.vocab_model import RemiVocab
from domain.constants.encoder.token_constants import BAR_KEY


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0), :]
        return x


class MLPPriorNetwork(nn.Module):
    """MLP-based prior network for generating prior distributions."""

    def __init__(self, input_dim, latent_dim, hidden_dim=512):
        super(MLPPriorNetwork, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        # Mean and log variance heads
        self.mean = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x):
        """
        Forward pass to compute the parameters of the prior distribution.

        Args:
            x: Input tensor

        Returns:
            mean: Mean of the distribution
            logvar: Log variance of the distribution
        """
        h = self.mlp(x)
        mean = self.mean(h)
        logvar = self.logvar(h)
        return mean, logvar


class MLPRecognitionNetwork(nn.Module):
    """MLP-based recognition network for generating posterior distributions."""

    def __init__(self, input_dim, latent_dim, hidden_dim=512):
        super(MLPRecognitionNetwork, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        # Mean and log variance heads
        self.mean = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x):
        """
        Forward pass to compute the parameters of the posterior distribution.

        Args:
            x: Input tensor

        Returns:
            mean: Mean of the distribution
            logvar: Log variance of the distribution
        """
        h = self.mlp(x)
        mean = self.mean(h)
        logvar = self.logvar(h)
        return mean, logvar


class TransformerEncoder(nn.Module):
    """Transformer encoder for VAE."""

    def __init__(self, input_dim, hidden_dim=512, num_layers=3, num_heads=8, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, src, src_mask=None):
        """
        Forward pass of the transformer encoder.

        Args:
            src: Source sequence [seq_len, batch_size, input_dim]
            src_mask: Mask for the source sequence

        Returns:
            output: Encoded sequence [seq_len, batch_size, hidden_dim]
        """
        # Project input to hidden dimension
        src = self.input_projection(src)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Apply transformer encoder
        output = self.transformer_encoder(src, src_mask)

        # Project to output dimension
        output = self.output_projection(output)

        return output


class TransformerDecoder(nn.Module):
    """Transformer decoder for VAE."""

    def __init__(self, output_dim, hidden_dim=512, num_layers=3, num_heads=8, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        # Input projection
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Forward pass of the transformer decoder.

        Args:
            tgt: Target sequence [seq_len, batch_size, hidden_dim]
            memory: Memory from encoder [seq_len, batch_size, hidden_dim]
            tgt_mask: Mask for the target sequence
            memory_mask: Mask for the memory sequence

        Returns:
            output: Decoded sequence [seq_len, batch_size, output_dim]
        """
        # Project input to hidden dimension
        tgt = self.input_projection(tgt)

        # Add positional encoding
        tgt = self.pos_encoder(tgt)

        # Apply transformer decoder
        output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask)

        # Project to output dimension
        output = self.output_projection(output)

        return output


class TransformerVAE(nn.Module):
    """
    Transformer VAE model for multimodal mapping.
    Processes the entire input sequence as a single unit.
    
    Now with specialized decoders:
    - Symbolic decoder conditioned on existing symbolic data
    - VQVAE decoder conditioned on existing VQVAE latent data
    """

    def __init__(self, input_dim=512, output_dim=256, hidden_dim=512, latent_dim=128, num_layers=4, num_heads=8, context_size=512, dropout=0.1):
        super(TransformerVAE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Use context_size for sequence length
        self.sequence_length = context_size

        # Encoder for processing input sequences
        self.encoder = TransformerEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout)

        # Latent variable networks
        self.prior = MLPPriorNetwork(input_dim=hidden_dim, latent_dim=self.latent_dim, hidden_dim=hidden_dim)
        self.recognition = MLPRecognitionNetwork(input_dim=hidden_dim * 2, latent_dim=self.latent_dim, hidden_dim=hidden_dim)

        # Reverse encoder for future context
        self.reverse_encoder = TransformerEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers // 2, num_heads=num_heads, dropout=dropout)

        # Specialized projections to condition on existing data
        self.symbolic_conditioning_projection = nn.Linear(output_dim // 2, hidden_dim)
        self.vqvae_conditioning_projection = nn.Linear(output_dim // 2, hidden_dim)

        # Decoder for symbolic features with specialized architecture
        self.symbolic_decoder = TransformerDecoder(
            output_dim=output_dim // 2, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout
        )

        # Decoder for VQVAE features with specialized architecture
        self.vqvae_decoder = TransformerDecoder(
            output_dim=output_dim // 2, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout
        )

        # Initial decoder input projection
        self.decoder_input_projection = nn.Linear(self.latent_dim, hidden_dim)
        
        # Additional processing for combined input
        self.symbolic_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.vqvae_fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, symbolic_input=None, vqvae_input=None, events=None, bar_ids=None, training=True):
        """
        Forward pass of the VAE with specialized decoders.

        Args:
            x: Input sequence [seq_len, batch_size, input_dim]
            symbolic_input: Optional symbolic features for conditioning [seq_len, batch_size, output_dim//2]
            vqvae_input: Optional VQVAE features for conditioning [seq_len, batch_size, output_dim//2]
            events: Not used in this implementation
            bar_ids: Not used in this implementation
            training: Whether in training mode

        Returns:
            symbolic_output: Output for symbolic features [seq_len, batch_size, output_dim//2]
            vqvae_output: Output for VQVAE features [seq_len, batch_size, output_dim//2]
            kl_loss: KL divergence loss
        """
        batch_size = x.size(1)
        seq_len = x.size(0)
        
        # If no conditioning inputs provided, create zero tensors
        if symbolic_input is None:
            symbolic_input = torch.zeros(seq_len, batch_size, self.output_dim // 2, device=x.device)
        
        if vqvae_input is None:
            vqvae_input = torch.zeros(seq_len, batch_size, self.output_dim // 2, device=x.device)
        
        # Encode the entire sequence
        encoded = self.encoder(x)
        
        # Calculate global representation by pooling across time dimension
        encoded_mean = encoded.mean(dim=0)  # [batch_size, hidden_dim]
        
        # Calculate prior distribution parameters
        prior_mean, prior_logvar = self.prior(encoded_mean)
        
        if training:
            # In training, calculate posterior with bidirectional context
            reverse_encoded = self.reverse_encoder(x.flip(0))
            reverse_encoded_mean = reverse_encoded.mean(dim=0)
            
            # Calculate posterior distribution parameters
            recog_input = torch.cat([encoded_mean, reverse_encoded_mean], dim=1)
            post_mean, post_logvar = self.recognition(recog_input)
            
            # Calculate KL divergence
            kl_loss = self.kl_divergence(post_mean, post_logvar, prior_mean, prior_logvar)
            
            # Sample from posterior
            z = self.reparameterize(post_mean, post_logvar)
        else:
            # In inference, sample from prior
            z = self.reparameterize(prior_mean, prior_logvar)
            kl_loss = torch.zeros(1, device=x.device)
        
        # Project latent variable to decoder dimension
        decoder_input = self.decoder_input_projection(z)
        decoder_input = decoder_input.unsqueeze(0)  # [1, batch_size, hidden_dim]
        
        # Create conditioning inputs for each decoder
        symbolic_conditioning = self.symbolic_conditioning_projection(symbolic_input)
        vqvae_conditioning = self.vqvae_conditioning_projection(vqvae_input)
        
        # Create specialized memory for each decoder by combining encoded sequence with conditioning
        symbolic_memory = encoded + symbolic_conditioning
        vqvae_memory = encoded + vqvae_conditioning
        
        # Decode using specialized decoders with conditioned memory
        symbolic_output = self.symbolic_decoder(tgt=decoder_input, memory=symbolic_memory)
        vqvae_output = self.vqvae_decoder(tgt=decoder_input, memory=vqvae_memory)
        
        # Expand the outputs to match the input sequence length if needed
        if symbolic_output.size(0) < seq_len:
            symbolic_output = symbolic_output.repeat(seq_len, 1, 1)
            vqvae_output = vqvae_output.repeat(seq_len, 1, 1)
        
        return symbolic_output, vqvae_output, kl_loss

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu: Mean of the distribution
            logvar: Log variance of the distribution

        Returns:
            z: Sampled latent code
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu1, logvar1, mu2, logvar2):
        """
        Compute KL divergence between two Gaussian distributions:
        KL(N(mu1, var1) || N(mu2, var2))

        Args:
            mu1, logvar1: Parameters of the first distribution
            mu2, logvar2: Parameters of the second distribution

        Returns:
            kl: KL divergence
        """
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)

        kl = 0.5 * torch.sum(logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / var2 - 1)

        return kl
