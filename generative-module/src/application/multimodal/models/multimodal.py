from lightning.pytorch import LightningModule
import torch
import torch.nn as nn
import torch.optim as optim
import random

from application.multimodal_mapping.models.input_encoders import TextEncoder, GlobalFeatureProcessor, MoodProcessor
from application.multimodal_mapping.models.image_encoder import ImageEncoder
from application.multimodal_mapping.models.embedding_fusion import CrossAttentionFusion, LinearConcatFusion
from application.multimodal_mapping.helpers.multimodal_mapping_helper import initialize_tokenizers_and_processors
from core.symbolic.models.vocab_model import SymbolicFeaturesVocab
from domain.constants.model_constants import ModelConstants


class MultimodalMappingModule(LightningModule):
    def __init__(
        self,
        # Input dimensions
        image_dim=384,
        text_dim=384,
        global_feature_dim=64,
        global_feature_out_dim=128,
        mood_dim=32,
        mood_out_dim=128,
        
        # Architecture parameters
        fusion_dim=512,
        d_model=512,
        latent_dim=128,
        output_dim=256,
        context_size=512,
        num_layers=4,
        num_heads=8,
        fusion_heads=8,
        fusion_type="linear_concat",
        dropout=0.1,
        training=True,
        
        # Modality dropout rates
        image_dropout_rate=0.2,
        text_dropout_rate=0.2,
        global_feature_dropout_rate=0.2,
        mood_dropout_rate=0.2,
        
        # Learning rate and schedule
        lr=5e-5,
        lr_schedule="cosine",
        warmup_steps=2000,
        max_steps=100000,
        
        # KL annealing
        use_kl_annealing=True,
        kl_start=0.0,
        kl_end=1.0,
        kl_anneal_steps=10000,
        
    ):
        super().__init__()
        
        # Save all parameters as hyperparameters for checkpointing
        self.save_hyperparameters()
        
        # Architecture parameters
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.global_feature_dim = global_feature_dim
        self.global_feature_out_dim = global_feature_out_dim
        self.mood_dim = mood_dim
        self.mood_out_dim = mood_out_dim
        self.fusion_dim = fusion_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.context_size = context_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.fusion_heads = fusion_heads
        self.fusion_type = fusion_type
        self.dropout = dropout
        self.training = training
        
        # Modality dropout rates
        self.image_dropout_rate = image_dropout_rate
        self.text_dropout_rate = text_dropout_rate
        self.global_feature_dropout_rate = global_feature_dropout_rate
        self.mood_dropout_rate = mood_dropout_rate
        
        # Learning rate and schedule parameters
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # KL annealing parameters
        self.use_kl_annealing = use_kl_annealing
        self.kl_start = kl_start
        self.kl_end = kl_end
        self.kl_anneal_steps = kl_anneal_steps
        self.kl_weight = kl_start
        
        # Learnable default embeddings for missing modalities
        self.default_image_embedding = nn.Parameter(torch.randn(image_dim))
        self.default_text_embedding = nn.Parameter(torch.randn(text_dim))
        self.default_global_embedding = nn.Parameter(torch.randn(global_feature_out_dim))
        self.default_mood_embedding = nn.Parameter(torch.randn(mood_out_dim))

        # Input encoders
        self.image_encoder = ImageEncoder(output_dim=image_dim)
        self.text_encoder = TextEncoder(output_dim=text_dim)

        # Create specialized processors for global features and moods
        self.global_feature_processor = GlobalFeatureProcessor(
            input_dim=global_feature_dim, 
            output_dim=global_feature_out_dim, 
            d_model=d_model, 
            dropout=dropout
        )

        self.mood_processor = MoodProcessor(
            input_dim=mood_dim, 
            output_dim=mood_out_dim, 
            d_model=d_model, 
            dropout=dropout
        )

        # Embedding fusion
        if fusion_type == "cross_attention":
            self.fusion_module = CrossAttentionFusion(
                embed_dim=fusion_dim, 
                num_heads=fusion_heads
            )
        else:
            self.fusion_module = LinearConcatFusion(
                input_dims=[image_dim, text_dim, global_feature_out_dim, mood_out_dim], 
                output_dim=fusion_dim
            )

        # TO BE IMPLEMENTED: VAE model
        self.vae = TransformerVAE(
            input_dim=fusion_dim,
            d_model=d_model,
            latent_dim=latent_dim,
            output_symbolic_dim=len(SymbolicFeaturesVocab()),
            output_vqvae_dim=ModelConstants.DEFAULT_D_LATENT,  # Use consistent default from constants
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # Initialize tokenizer for text prompts
        self.tokenizer, self.image_processor = initialize_tokenizers_and_processors()

    def _apply_modality_dropout(self, has_image, has_text, has_global, has_mood):
        """
        Apply modality dropout during training to help the model learn
        to handle missing modalities.
        
        Args:
            has_image: Whether image modality is available
            has_text: Whether text modality is available
            has_global: Whether global feature modality is available
            has_mood: Whether mood modality is available
            
        Returns:
            Tuple of booleans indicating which modalities to use
        """
        if not self.training:
            # During inference, use whatever is available
            return has_image, has_text, has_global, has_mood
            
        # During training, randomly drop modalities
        use_image = has_image and (random.random() > self.image_dropout_rate)
        use_text = has_text and (random.random() > self.text_dropout_rate)
        use_global = has_global and (random.random() > self.global_feature_dropout_rate)
        use_mood = has_mood and (random.random() > self.mood_dropout_rate)
        
        # Ensure at least one modality is used
        if not (use_image or use_text or use_global or use_mood):
            # If all would be dropped, keep one randomly
            modality_idx = random.randint(0, 3)
            if modality_idx == 0 and has_image:
                use_image = True
            elif modality_idx == 1 and has_text:
                use_text = True
            elif modality_idx == 2 and has_global:
                use_global = True
            elif has_mood:
                use_mood = True
            else:
                # If nothing is available, use whatever is available
                use_image = has_image
                use_text = has_text
                use_global = has_global
                use_mood = has_mood
                
        return use_image, use_text, use_global, use_mood

    def forward(self, batch):
        # Determine which modalities are available in the batch
        has_image = "images" in batch
        has_text = "text_prompts" in batch
        has_global = "global_features" in batch
        has_mood = "moods" in batch
        
        # Apply modality dropout during training
        use_image, use_text, use_global, use_mood = self._apply_modality_dropout(has_image, has_text, has_global, has_mood)
        
        # Initialize embeddings
        batch_size = next(iter(batch.values())).shape[0]
        device = next(iter(batch.values())).device
        
        # Process image input if available and selected
        if use_image and has_image:
            image_embedding = self.image_encoder(batch["images"])
        else:
            # Use default embedding expanded to batch size
            image_embedding = self.default_image_embedding.unsqueeze(0).expand(batch_size, -1).to(device)

        # Process text input if available and selected
        if use_text and has_text:
            # Tokenize text prompts
            encoded_text = self.tokenizer(batch["text_prompts"], padding="max_length", truncation=True, max_length=ModelConstants.MAX_TEXT_LENGTH, return_tensors="pt").to(device)
            text_embedding = self.text_encoder(input_ids=encoded_text["input_ids"], attention_mask=encoded_text["attention_mask"])
        else:
            # Use default embedding expanded to batch size
            text_embedding = self.default_text_embedding.unsqueeze(0).expand(batch_size, -1).to(device)

        # Process global features if available and selected
        if use_global and has_global:
            global_features = batch["global_features"]
            global_embedding = self.global_feature_processor(global_features)
        else:
            # Use default embedding expanded to batch size
            global_embedding = self.default_global_embedding.unsqueeze(0).expand(batch_size, -1).to(device)

        # Process moods if available and selected
        if use_mood and has_mood:
            moods = batch["moods"]
            mood_embedding = self.mood_processor(moods)
        else:
            # Use default embedding expanded to batch size
            mood_embedding = self.default_mood_embedding.unsqueeze(0).expand(batch_size, -1).to(device)

        # Prepare embeddings and their availability flags for fusion
        embeddings = {
            "image": {"embedding": image_embedding, "available": use_image and has_image},
            "text": {"embedding": text_embedding, "available": use_text and has_text},
            "global": {"embedding": global_embedding, "available": use_global and has_global},
            "mood": {"embedding": mood_embedding, "available": use_mood and has_mood}
        }

        # Fuse embeddings based on what's available
        if isinstance(self.fusion_module, CrossAttentionFusion):
            # For cross-attention, we'll use the first available embedding as query
            # and the rest as key-value pairs
            available_embeddings = [emb_info["embedding"] for emb_name, emb_info in embeddings.items() 
                                   if emb_info["available"]]
            
            if not available_embeddings:
                # If no embeddings are available, use all defaults
                available_embeddings = [image_embedding, text_embedding, global_embedding, mood_embedding]
            
            # Use the first available embedding as query, rest as key-value
            query_embed = available_embeddings[0]
            key_value_embeds = available_embeddings[1:] if len(available_embeddings) > 1 else [available_embeddings[0]]
            
            fused_embedding = self.fusion_module(query_embed=query_embed, key_value_embeds=key_value_embeds)
        else:  # LinearConcatFusion
            # For linear concat, we use all embeddings (including defaults for missing ones)
            fused_embedding = self.fusion_module(image_embedding, text_embedding, global_embedding, mood_embedding)

        # Convert fused embedding to correct shape for transformer [seq_len, batch_size, input_dim]
        # For now, we create a sequence of length 1 by adding a dimension
        fused_embedding = fused_embedding.unsqueeze(1)  # [batch_size, 1, input_dim]

        # Get the existing symbolic and VQVAE features for conditioning if available
        symbolic_conditioning = batch.get("bar_symbolic", None)
        vqvae_conditioning = batch.get("latents", None)

        # Process the fused embedding through the VAE model with conditioning
        symbolic_features, vqvae_features, kl_loss = self.vae(
            x=fused_embedding, 
            symbolic_input=symbolic_conditioning, 
            vqvae_input=vqvae_conditioning, 
            training=self.training
        )

        return {"symbolic_features": symbolic_features, "vqvae_features": vqvae_features, "kl_loss": kl_loss}

    def get_kl_weight(self):
        if not self.use_kl_annealing or self.global_step >= self.kl_anneal_steps:
            return self.kl_end

        # Linear annealing
        kl_weight = self.kl_start + (self.kl_end - self.kl_start) * (self.global_step / self.kl_anneal_steps)
        return kl_weight

    def _calculate_loss(self, outputs, batch):
        """
        Calculate loss components and total loss.
        Shared between training, validation, and test steps.

        Args:
            outputs: Dictionary with model outputs
            batch: Input batch

        Returns:
            Dictionary with loss components and total loss
        """
        # Reconstruction loss
        symbolic_loss = torch.nn.functional.mse_loss(outputs["symbolic_features"], batch["bar_symbolic"])

        vqvae_loss = torch.nn.functional.mse_loss(outputs["vqvae_features"], batch["latents"])

        # Apply KL weight - use the self.kl_weight set by callback
        kl_loss = outputs["kl_loss"] * self.kl_weight

        # Total loss
        loss = symbolic_loss + vqvae_loss + kl_loss

        return {
            "loss": loss,
            "symbolic_loss": symbolic_loss,
            "vqvae_loss": vqvae_loss,
            "kl_loss": kl_loss,
        }

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        losses = self._calculate_loss(outputs, batch)
        print(outputs, flush=True)

        # Log metrics
        self.log("train_loss", losses["loss"], prog_bar=True)
        self.log("symbolic_loss", losses["symbolic_loss"])
        self.log("vqvae_loss", losses["vqvae_loss"])
        self.log("kl_loss", losses["kl_loss"])
        self.log("kl_weight", self.kl_weight)

        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)

        # For validation, always use final KL weight
        self.kl_weight = self.kl_end

        losses = self._calculate_loss(outputs, batch)

        # Log metrics
        self.log("valid_loss", losses["loss"], prog_bar=True)
        self.log("valid_symbolic_loss", losses["symbolic_loss"])
        self.log("valid_vqvae_loss", losses["vqvae_loss"])
        self.log("valid_kl_loss", losses["kl_loss"])

        return losses["loss"]

    def test_step(self, batch, batch_idx):
        """
        Test step for evaluation.

        Args:
            batch: Input batch
            batch_idx: Batch index

        Returns:
            Test loss
        """
        outputs = self.forward(batch)

        # For testing, always use final KL weight
        self.kl_weight = self.kl_end

        losses = self._calculate_loss(outputs, batch)

        # Log metrics
        self.log("test_loss", losses["loss"])
        self.log("test_symbolic_loss", losses["symbolic_loss"])
        self.log("test_vqvae_loss", losses["vqvae_loss"])
        self.log("test_kl_loss", losses["kl_loss"])

        # Calculate additional metrics like feature similarity
        similarity_score = self._calculate_similarity(outputs["symbolic_features"], batch["target_symbolic_features"])
        self.log("feature_similarity", similarity_score)

        return losses["loss"]

    def _calculate_similarity(self, pred_features, target_features):
        """
        Calculate cosine similarity between predicted and target features.

        Args:
            pred_features: Predicted features
            target_features: Target features

        Returns:
            Average cosine similarity
        """
        # Normalize features
        pred_norm = torch.nn.functional.normalize(pred_features, dim=-1)
        target_norm = torch.nn.functional.normalize(target_features, dim=-1)

        # Calculate cosine similarity
        similarity = torch.sum(pred_norm * target_norm, dim=-1).mean()

        return similarity

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        if self.lr_schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_steps - self.warmup_steps)
        elif self.lr_schedule == "linear":
            scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.max_steps - self.warmup_steps)
        else:  # constant
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        # Warmup scheduler
        if self.warmup_steps > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.warmup_steps)

            # Chain schedulers
            from torch.optim.lr_scheduler import SequentialLR

            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[self.warmup_steps])

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class TransformerVAE(nn.Module):
    """
    A Transformer-based Variational Autoencoder that maps fused multimodal embeddings
    to both symbolic features and VQVAE latent features.
    """
    def __init__(
        self,
        input_dim=512,
        d_model=512,
        latent_dim=128,
        output_symbolic_dim=1024,
        output_vqvae_dim=1024,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        # Save dimensions
        self.input_dim = input_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.output_symbolic_dim = output_symbolic_dim
        self.output_vqvae_dim = output_vqvae_dim
        
        # Encoder: Transform input embeddings into hidden representation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # VAE components: Map symbolic output to latent distribution parameters
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        
        # Decoder: Map from latent space to output dimensions
        self.fc_latent_to_hidden = nn.Linear(latent_dim, input_dim)
        
        # Symbolic features decoder
        symbolic_decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True
        )
        self.symbolic_decoder = nn.TransformerDecoder(
            symbolic_decoder_layer,
            num_layers=num_layers
        )
        self.symbolic_projection = nn.Linear(input_dim, output_symbolic_dim)
        
        # VQVAE latent features decoder
        vqvae_decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True
        )
        self.vqvae_decoder = nn.TransformerDecoder(
            vqvae_decoder_layer,
            num_layers=num_layers
        )
        self.vqvae_projection = nn.Linear(input_dim, output_vqvae_dim)

    def encode(self, x):
        """
        Encode input sequence to latent distribution parameters.
        
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            
        Returns:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        # Encode input sequence
        encoded = self.encoder(x)
        
        # Get sequence representation (use mean pooling across sequence dimension)
        encoded_pooled = encoded.mean(dim=1)  # Average over sequence length (dim=1 for batch_first=True)
        
        # Map to latent distribution parameters
        mu = self.fc_mu(encoded_pooled)
        logvar = self.fc_logvar(encoded_pooled)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar, training=None):
        """
        Reparameterization trick: sample from the latent distribution.
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            training: Whether in training mode (if None, uses self.training)
            
        Returns:
            Sampled latent vector
        """
        if training is None:
            training = self.training
            
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # During inference, use the mean directly
            z = mu
            
        return z
    
    def decode(self, z, symbolic_input=None, vqvae_input=None):
        """
        Decode latent vector to symbolic and VQVAE features.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            symbolic_input: Optional conditioning symbolic features [batch_size, seq_len, dim] or None
            vqvae_input: Optional conditioning VQVAE features [batch_size, seq_len, dim] or None
            
        Returns:
            symbolic_output: Reconstructed symbolic features [batch_size, output_symbolic_dim]
            vqvae_output: Reconstructed VQVAE features [batch_size, output_vqvae_dim]
        """
        # Transform latent vector to initial hidden state
        hidden = self.fc_latent_to_hidden(z)
        
        # Create memory for decoders 
        memory = hidden.unsqueeze(1)  # [batch_size, 1, input_dim] - for batch_first=True
        
        # If no conditioning inputs are provided, create empty tensors
        batch_size = z.size(0)
        device = z.device
        
        # Create properly shaped conditioning inputs if not provided
        if symbolic_input is None:
            # Create learnable or zero tensor for symbolic decoder with correct dimensions
            symbolic_input = torch.zeros((batch_size, 1, self.input_dim), device=device)
        elif symbolic_input.size(-1) != self.input_dim:
            # If symbolic_input has wrong feature dimension, project it
            projection = nn.Linear(symbolic_input.size(-1), self.input_dim).to(device)
            symbolic_input = projection(symbolic_input)
        
        if vqvae_input is None:
            # Create learnable or zero tensor for VQVAE decoder with correct dimensions
            vqvae_input = torch.zeros((batch_size, 1, self.input_dim), device=device)
        elif vqvae_input.size(-1) != self.input_dim:
            # If vqvae_input has wrong feature dimension, project it
            projection = nn.Linear(vqvae_input.size(-1), self.input_dim).to(device)
            vqvae_input = projection(vqvae_input)
        
        # Generate symbolic features
        symbolic_output = self.symbolic_decoder(
            tgt=symbolic_input, 
            memory=memory
        )
        symbolic_features = self.symbolic_projection(symbolic_output)
        
        # Generate VQVAE latent features
        vqvae_output = self.vqvae_decoder(
            tgt=vqvae_input,
            memory=memory
        )
        vqvae_features = self.vqvae_projection(vqvae_output)
        
        return symbolic_features, vqvae_features
    
    def forward(self, x, symbolic_input=None, vqvae_input=None, training=None):
        """
        Forward pass through the VAE.
        
        Args:
            x: Input sequence [batch_size, seq_len, input_dim] 
            symbolic_input: Optional conditioning symbolic features [batch_size, seq_len, dim]
            vqvae_input: Optional conditioning VQVAE features [batch_size, seq_len, dim]
            training: Whether in training mode (if None, uses self.training)
            
        Returns:
            symbolic_features: Reconstructed symbolic features
            vqvae_features: Reconstructed VQVAE features
            kl_loss: KL divergence loss term
        """
        # Set training mode
        if training is None:
            training = self.training
        
        # Ensure x is in the expected format [batch_size, seq_len, input_dim]
        if x.dim() == 3 and x.size(0) != x.size(1):
            # Input shape is correct, no need to transpose
            pass
            
        # Encode input to latent distribution parameters
        mu, logvar = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar, training)
        
        # Handle conditioning inputs - ensure they have the expected shape
        # Create input tensors with the right dimensions if needed
        batch_size = z.size(0)
        device = z.device
        
        # For conditioning inputs, we'll create projection layers if needed
        # to match the expected input dimensions for the decoders
        if symbolic_input is not None:
            # Ensure symbolic_input has the correct shape [batch_size, seq_len, dim]
            if symbolic_input.dim() == 2:
                # Add sequence dimension
                symbolic_input = symbolic_input.unsqueeze(1)
        
        if vqvae_input is not None:
            # Ensure vqvae_input has the correct shape [batch_size, seq_len, dim]
            if vqvae_input.dim() == 2:
                # Add sequence dimension
                vqvae_input = vqvae_input.unsqueeze(1)
        
        # Decode latent vector to outputs
        symbolic_features, vqvae_features = self.decode(z, symbolic_input, vqvae_input)
        
        # Calculate KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        
        return symbolic_features, vqvae_features, kl_loss
