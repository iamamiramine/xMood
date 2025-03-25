from lightning.pytorch import LightningModule
import torch
import torch.optim as optim

from application.multimodal_mapping.models.input_encoders import TextEncoder, GlobalFeatureProcessor, MoodProcessor
from application.multimodal_mapping.models.embedding_fusion import CrossAttentionFusion, LinearConcatFusion
from application.multimodal_mapping.models.hierarchical_vae import TransformerVAE
from application.multimodal_mapping.helpers.multimodal_mapping_helper import initialize_tokenizers_and_processors


class MultimodalMappingModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Get dimensions from config
        text_dim = self.config.get("text_dim", 384)

        # Define separate dimensions for each input type
        global_feature_dim = self.config.get("global_feature_dim", 64)
        global_feature_out_dim = self.config.get("global_feature_out_dim", 128)

        mood_dim = self.config.get("mood_dim", 32)
        mood_out_dim = self.config.get("mood_out_dim", 128)

        fusion_dim = self.config.get("fusion_dim", 512)
        hidden_dim = self.config.get("hidden_dim", 512)
        latent_dim = self.config.get("latent_dim", 128)  # Updated parameter name
        output_dim = self.config.get("output_dim", 256)
        context_size = self.config.get("context_size", 512)  # Use context_size instead of max_bars/sequence_length
        self.training = self.config.get("training", True)

        # Input encoders
        self.text_encoder = TextEncoder(output_dim=text_dim)

        # Create specialized processors for global features and moods with improved architecture
        self.global_feature_processor = GlobalFeatureProcessor(
            input_dim=global_feature_dim, output_dim=global_feature_out_dim, hidden_dim=hidden_dim, dropout=self.config.get("dropout", 0.1)
        )

        self.mood_processor = MoodProcessor(input_dim=mood_dim, output_dim=mood_out_dim, hidden_dim=hidden_dim, dropout=self.config.get("dropout", 0.1))

        # Embedding fusion
        fusion_type = self.config.get("fusion_type", "linear_concat")
        if fusion_type == "cross_attention":
            self.fusion_module = CrossAttentionFusion(embed_dim=fusion_dim, num_heads=self.config.get("fusion_heads", 8))
        else:
            self.fusion_module = LinearConcatFusion(input_dims=[text_dim, global_feature_out_dim, mood_out_dim], output_dim=fusion_dim)

        # VAE model (renamed from hierarchical_vae to vae)
        self.vae = TransformerVAE(
            input_dim=fusion_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,  # Updated parameter name
            num_layers=self.config.get("num_layers", 4),
            num_heads=self.config.get("num_heads", 8),
            context_size=context_size,  # Use context_size instead of max_bars
            dropout=self.config.get("dropout", 0.1),
        )

        # Initialize tokenizer for text prompts
        self.tokenizer = initialize_tokenizers_and_processors()[0]  # Get only the text tokenizer

        # Learning rate and schedule
        self.lr = self.config.get("lr", 5e-5)
        self.lr_schedule = self.config.get("lr_schedule", "cosine")
        self.warmup_steps = self.config.get("warmup_steps", 2000)
        self.max_steps = self.config.get("max_steps", 100000)

        # KL annealing
        self.use_kl_annealing = self.config.get("use_kl_annealing", True)
        self.kl_start = self.config.get("kl_start", 0.0)
        self.kl_end = self.config.get("kl_end", 1.0)
        self.kl_anneal_steps = self.config.get("kl_anneal_steps", 10000)

        # Current KL weight (will be updated by callback during training)
        self.kl_weight = self.kl_start

    def forward(self, batch):
        # Process inputs based on what's available in the batch

        # Process text input - either from text_prompts or input_ids
        if "text_prompts" in batch:
            # Tokenize text prompts
            encoded_text = self.tokenizer(batch["text_prompts"], padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(self.device)
            text_embedding = self.text_encoder(input_ids=encoded_text["input_ids"], attention_mask=encoded_text["attention_mask"])

        # Process global features if available
        if "global_features" in batch:
            global_features = batch["global_features"]
            global_embedding = self.global_feature_processor(global_features)

        # Process moods if available
        if "moods" in batch:
            moods = batch["moods"]
            mood_embedding = self.mood_processor(moods)

        # Fuse embeddings - use all available embeddings together
        if isinstance(self.fusion_module, CrossAttentionFusion):
            # Use text as query embedding, others as key-value embeddings
            fused_embedding = self.fusion_module(query_embed=text_embedding, key_value_embeds=[global_embedding, mood_embedding])
        else:  # LinearConcatFusion
            fused_embedding = self.fusion_module(text_embedding, global_embedding, mood_embedding)

        # Convert fused embedding to correct shape for transformer [seq_len, batch_size, input_dim]
        # For now, we create a sequence of length 1 by adding a dimension
        fused_embedding = fused_embedding.unsqueeze(0)  # [1, batch_size, input_dim]

        # Get the existing symbolic and VQVAE features for conditioning if available
        symbolic_conditioning = batch.get("bar_symbolic", None)
        vqvae_conditioning = batch.get("latents", None)

        # Process the fused embedding through the VAE model with conditioning
        symbolic_features, vqvae_features, kl_loss = self.vae(
            x=fused_embedding, symbolic_input=symbolic_conditioning, vqvae_input=vqvae_conditioning, training=self.training
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
        symbolic_loss = torch.nn.functional.mse_loss(outputs["symbolic_features"], batch["target_symbolic_features"])

        vqvae_loss = torch.nn.functional.mse_loss(outputs["vqvae_features"], batch["target_vqvae_features"])

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
