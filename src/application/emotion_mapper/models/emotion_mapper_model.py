"""
This module implements an emotion mapping model that translates musical features and symbolic features into bar-level emotion predictions.
The model uses a transformer-based architecture to map latent representations and textual symbolic features to a sequence of emotion vectors.
It employs self-attention mechanisms to learn temporal relationships and emotional context across musical bars.
"""

# Standard PyTorch imports
import torch
from torch import nn
import torch.nn.functional as F

# PyTorch Lightning for training framework
import lightning.pytorch as pl

# PyTorch optimization
from torch.optim import Adam

# PyTorch utilities
from torch.nn.utils.rnn import pad_sequence

from src.application.encoder.helpers.vocab_helper import get_bars, get_bos_eos_events

# Local application imports
from src.application.encoder.models.vocab_model import SymbolicFeaturesVocab, RemiVocab
from src.domain.constants.encoder.token_constants import BAR_KEY


def get_embedding(tensor, embedding_fn, device):
    """
    Apply an embedding function to a tensor after moving it to the specified device.

    Args:
        tensor (torch.Tensor): Input tensor to embed, or None
        embedding_fn (callable): Function to apply embedding
        device (torch.device): Device to move tensor to

    Returns:
        torch.Tensor or None: Embedded tensor if input is not None, otherwise None
    """
    if tensor is not None:
        return embedding_fn(tensor.to(device))
    return None


def pad_and_transpose(embeddings):
    """
    Pad and transpose a list of embeddings to create a uniform tensor.

    Args:
        embeddings (list): List of embedding tensors

    Returns:
        torch.Tensor: Padded and transposed tensor of shape [batch_size, features, seq_len]
    """
    transposed = [emb.transpose(0, 1) for emb in embeddings if emb is not None]
    padded = pad_sequence(transposed, batch_first=True)
    return padded.transpose(1, 2)


def concatenate_embeddings(*embeddings):
    """
    Concatenate multiple embeddings along the last dimension, filtering out None values.

    Args:
        *embeddings: Variable number of embedding tensors

    Returns:
        torch.Tensor: Concatenated embeddings
    """
    return torch.cat([emb for emb in embeddings if emb is not None], dim=-1)


class EmotionMapper(pl.LightningModule):
    """
    A PyTorch Lightning module that maps musical features and symbolic features to bar-level emotion predictions.

    The model uses a transformer-based architecture with:
    - Bar and position embeddings to capture temporal information
    - Multi-head self-attention to learn relationships between bars
    - Emotion projection and decoding layers
    - Contrastive learning with temporal regularization

    Args:
        feature_dim (int): Dimension of input feature vectors
        d_model (int, optional): Hidden dimension of the model. Defaults to 512
        num_heads (int, optional): Number of attention heads. Defaults to 8
        max_bars (int, optional): Maximum number of bars to process. Defaults to 512
    """

    def __init__(self, feature_dim, d_model=512, num_heads=8, max_bars=512, max_positions=1024):
        super().__init__()
        self.d_model = d_model  # model dimension
        self.latent_dim = feature_dim  # latent dimension
        self.max_bars = max_bars  # maximum number of bars
        self.max_positions = max_positions  # maximum number of positions
        self.context_size = -1  # Matches generator model context size

        desc_vocab = SymbolicFeaturesVocab()  # description vocabulary

        # Embeddings for positional information
        self.bar_embedding = nn.Embedding(max_bars + 1, d_model)  # Embeds bar position
        self.pos_embedding = nn.Embedding(max_positions + 1, d_model)  # Embeds position within bar

        # Input projections
        self.latent_in = nn.Linear(self.latent_dim, self.d_model, bias=False)  # Projects latent features
        self.desc_in = nn.Embedding(len(desc_vocab), self.d_model)  # Embeds description tokens
        self.emotion_projector = nn.Sequential(nn.Linear(5, d_model), nn.LayerNorm(d_model))  # Projects 5-dimensional emotion vectors to model dimension

        # Emotion decoder - maps from model dimension back to 5 emotion probabilities
        self.emotion_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 5), nn.Softmax(dim=-1)  # Ensures outputs sum to 1
        )

        # Attention mechanism components
        self.key_proj = nn.Linear(d_model, d_model)  # Projects inputs to key space
        self.value_proj = nn.Linear(d_model, d_model)  # Projects inputs to value space
        self.self_attention = nn.MultiheadAttention(d_model, num_heads)  # Multi-head attention
        self.norm = nn.LayerNorm(d_model)  # Layer normalization

        # Projects concatenated description features
        self.desc_proj = nn.Linear(2 * d_model, d_model, bias=False)

        # Loss function hyperparameters
        self.margin = 0.3  # Margin for contrastive learning
        self.lambda_reg = 0.5  # Weight for temporal regularization
        self.lambda_l2 = 0.1  # Weight for L2 regularization

        # List of emotion categories
        self.emotions = ["anger", "joy", "love", "sadness", "surprise"]

    def encode_features(self, z=None, desc_bar_ids=None, bar_ids=None, position_ids=None):
        """
        Encodes input features (latents and/or symbolic features) with positional information.

        This method processes different types of input features:
        1. Latent vectors from the music representation
        2. Symbolic Features tokens from textual symbolic features
        3. Bar and position information for temporal context

        Args:
            z (dict, optional): Dictionary containing:
                - latents (torch.Tensor): Latent vectors [batch_size, seq_len, latent_dim]
                - description (torch.Tensor): Symbolic Features token indices [batch_size, desc_len]
            desc_bar_ids (torch.Tensor, optional): Bar IDs for description tokens [batch_size, desc_len]
            bar_ids (torch.Tensor, optional): Bar IDs for the sequence [batch_size, seq_len]
            position_ids (torch.Tensor, optional): Position IDs within bars [batch_size, seq_len]

        Returns:
            torch.Tensor or None: Encoded features with shape [batch_size, seq_len, d_model]
        """
        # Early return if no input features provided
        if z is None:
            return None

        # Initialize embeddings for description and latent features
        desc_emb, latent_emb = None, None

        # Process description tokens if present in input
        if "description" in z:
            # Convert description tokens to embeddings and move to correct device
            desc_emb = get_embedding(z.get("description"), self.desc_in, self.device)

        # Process latent vectors if present in input
        if "latents" in z:
            # Project latent vectors to model dimension and move to correct device
            latent_emb = get_embedding(z.get("latents"), self.latent_in, self.device)

        # Collect all non-None embeddings into a list
        embeddings = [emb for emb in [desc_emb, latent_emb] if emb is not None]

        # Return None if no valid embeddings were created
        if not embeddings:
            return None

        # Pad and transpose embeddings to create uniform tensor
        padded_embeddings = pad_and_transpose(embeddings)

        if len(padded_embeddings) == 2:
            desc_emb, latent_emb = padded_embeddings
        elif len(padded_embeddings) == 1:
            if desc_emb is not None:
                desc_emb = padded_embeddings[0]
            elif latent_emb is not None:
                latent_emb = padded_embeddings[0]

        z_embeddings = [emb for emb in [desc_emb, latent_emb] if emb is not None]

        # Process embeddings based on how many are available
        if len(z_embeddings) == 1:
            # If only one embedding type, use it directly
            z_emb = embeddings[0]
            # Add bar information to description embedding if available
            if desc_bar_ids is not None and desc_emb is not None:
                z_emb += self.bar_embedding(desc_bar_ids.to(self.device))
        elif len(z_embeddings) > 1:
            # If multiple embeddings present
            if desc_bar_ids is not None and desc_emb is not None:
                # Add bar information to description embedding
                desc_emb = desc_emb + self.bar_embedding(desc_bar_ids.to(self.device))
                # Update embeddings list with modified description embedding
                z_embeddings = [emb for emb in [desc_emb, latent_emb] if emb is not None]
            # Concatenate all embeddings and project to model dimension
            z_emb = self.desc_proj(concatenate_embeddings(*z_embeddings))

        if bar_ids is not None:
            bar_ids = torch.clamp(bar_ids, max=self.max_bars)
            # Ensure bar_ids matches z_emb's sequence length
            if bar_ids.size(1) != z_emb.size(1):
                # Truncate or pad bar_ids to match z_emb's sequence length
                if bar_ids.size(1) > z_emb.size(1):
                    bar_ids = bar_ids[:, : z_emb.size(1)]
                else:
                    # Pad with zeros if bar_ids is shorter
                    pad_size = z_emb.size(1) - bar_ids.size(1)
                    bar_ids = F.pad(bar_ids, (0, pad_size), value=0)
            z_emb += self.bar_embedding(bar_ids)
        if position_ids is not None:
            position_ids = torch.clamp(position_ids, max=self.max_positions)
            # Ensure position_ids matches z_emb's sequence length
            if position_ids.size(1) != z_emb.size(1):
                # Truncate or pad position_ids to match z_emb's sequence length
                if position_ids.size(1) > z_emb.size(1):
                    position_ids = position_ids[:, : z_emb.size(1)]
                else:
                    # Pad with zeros if position_ids is shorter
                    pad_size = z_emb.size(1) - position_ids.size(1)
                    position_ids = F.pad(position_ids, (0, pad_size), value=0)
            z_emb += self.pos_embedding(position_ids)

        return z_emb  # Return final encoded features

    def encode_emotions(self, z):
        """
        Encodes emotion vectors into the model's hidden dimension space.

        Args:
            z (dict): Dictionary containing:
                - emotions_vector (torch.Tensor): Emotion scores [batch_size, 5]

        Returns:
            torch.Tensor or None: Encoded emotions with shape [batch_size, d_model]
        """
        # Early return if input is None or doesn't contain emotions
        if z is None or "emotions_vector" not in z:
            return None

        # Move emotion vector to correct device and project to model dimension
        emotions_vector = z["emotions_vector"].to(self.device)  # Shape: [batch_size, 5]
        # Transform 5-dimensional emotion vector to d_model dimension and apply layer normalization
        emotion_embedding = self.emotion_projector(emotions_vector)  # Shape: [batch_size, d_model]

        # Create list of valid embeddings (filtering out None values)
        embeddings = [emb for emb in [emotion_embedding] if emb is not None]
        # Return None if no valid embeddings exist
        if not embeddings:
            return None

        # Pad and transpose embeddings to create uniform tensor
        # This step ensures consistent dimensionality across different batch sizes
        padded_embeddings = pad_and_transpose(embeddings)  # Shape: [batch_size, d_model, seq_len]

        # Extract emotion embedding from padded tensor if present
        if len(padded_embeddings) == 1:
            emotion_emb = padded_embeddings[0]  # Shape: [batch_size, d_model, 1]

        # Collect final emotion embeddings for processing
        z_embeddings = [emb for emb in [emotion_emb] if emb is not None]
        # If we have exactly one embedding, use it directly
        if len(z_embeddings) == 1:
            z_emb = embeddings[0]  # Shape: [batch_size, d_model]

        # Return the final encoded emotion representation
        return z_emb  # Shape: [batch_size, d_model]

    def forward(self, z, desc_bar_ids=None, bar_ids=None, position_ids=None):
        """
        Forward pass of the emotion mapper model.

        This method:
        1. Encodes input features and emotions
        2. Applies self-attention to distribute emotions across bars
        3. Generates bar-level emotion predictions

        Args:
            z (dict): Dictionary containing input features and emotions
            desc_bar_ids (torch.Tensor, optional): Bar IDs for description
            bar_ids (torch.Tensor, optional): Bar IDs for the sequence
            position_ids (torch.Tensor, optional): Position IDs within bars

        Returns:
            dict: Dictionary containing:
                - predictions (torch.Tensor): Bar-level emotion predictions [batch_size, seq_len, 5]
                - raw_emotions (torch.Tensor): Input emotion vectors [batch_size, 5]
                - attention_weights (torch.Tensor): Attention weights [batch_size, num_heads, seq_len, seq_len]

        Raises:
            ValueError: If either bar features or emotion features are missing
        """
        # Step 1: Encode input features and emotions
        # Transform bar-level features (latents/symbolic features) into model dimension
        bar_embeddings = self.encode_features(z, desc_bar_ids, bar_ids, position_ids)  # Shape: [batch_size, seq_len, d_model]
        # Transform piece-level emotions into model dimension
        piece_embedding = self.encode_emotions(z)  # Shape: [batch_size, d_model]

        # Validate that both required inputs are present
        if bar_embeddings is None or piece_embedding is None:
            raise ValueError("Both bar features and emotion features are required")

        # Step 2: Prepare inputs for attention mechanism
        # Project bar embeddings to create queries (each bar asks how much emotion it should get)
        queries = self.key_proj(bar_embeddings).transpose(0, 1)  # Shape: [seq_len, batch_size, d_model]

        # Project piece emotions to create keys (determine relevance of emotions to each bar)
        keys = self.key_proj(piece_embedding).unsqueeze(0)  # Shape: [1, batch_size, d_model]

        # Project piece emotions to create values (actual emotion content to be distributed)
        values = self.value_proj(piece_embedding).unsqueeze(0)  # Shape: [1, batch_size, d_model]

        # Step 3: Apply attention mechanism
        # Use multi-head attention to distribute emotions across bars based on relevance
        attended, attention_weights = self.self_attention(
            query=queries, key=keys, value=values  # What each bar is asking for  # What emotions are available  # The actual emotion content
        )  # attended shape: [seq_len, batch_size, d_model]
        # attention_weights shape: [batch_size, num_heads, seq_len, 1]

        # Apply dropout to attention weights for regularization
        attention_weights = F.dropout(attention_weights, p=0.1, training=self.training)

        # Step 4: Process attention outputs
        # Reshape attended features back to batch-first format
        attended = attended.transpose(0, 1)  # Shape: [batch_size, seq_len, d_model]
        # Reshape bar embeddings back to batch-first format
        bar_embeddings = queries.transpose(0, 1)  # Shape: [batch_size, seq_len, d_model]
        # Concatenate attended emotions with original bar features
        combined = torch.cat([attended, bar_embeddings], dim=-1)  # Shape: [batch_size, seq_len, 2*d_model]

        # Step 5: Generate final predictions
        # Project combined features back to model dimension
        bar_predictions = self.desc_proj(combined)  # Shape: [batch_size, seq_len, d_model]
        # Apply layer normalization
        bar_predictions = self.norm(bar_predictions)  # Shape: [batch_size, seq_len, d_model]
        # Decode to emotion probabilities
        emotion_predictions = self.emotion_decoder(bar_predictions)  # Shape: [batch_size, seq_len, 5]

        # Return predictions and intermediate results
        return {
            "predictions": emotion_predictions,  # Bar-level emotion predictions
            "raw_emotions": z["emotions_vector"],  # Original input emotions
            "attention_weights": attention_weights,  # Attention distribution
        }

    def compute_losses(self, bar_predictions, piece_emotion, attention_weights):
        """
        Compute the combined loss for training the emotion mapper.

        The loss consists of three components:
        1. Alignment loss: Ensures predictions match the piece-level emotions
        2. Temporal loss: Encourages smooth transitions between consecutive bars
        3. L2 regularization: Prevents predictions from growing too large

        Args:
            bar_predictions (torch.Tensor): Predicted emotions per bar [batch_size, seq_len, 5]
            piece_emotion (torch.Tensor): Ground truth piece-level emotions [batch_size, 5]
            attention_weights (torch.Tensor): Attention weights [batch_size, num_heads, seq_len, seq_len]

        Returns:
            dict: Dictionary containing:
                - total_loss (torch.Tensor): Combined loss value
                - align_loss (torch.Tensor): Alignment loss component
                - temp_loss (torch.Tensor): Temporal regularization loss
                - reg_loss (torch.Tensor): L2 regularization loss
        """
        # Expand piece emotions to match the shape of bar predictions by:
        # 1. Adding sequence dimension (unsqueeze)
        # 2. Repeating emotions for each bar position (expand)
        # Result shape: [batch_size, seq_len, 5]
        piece_emb_expanded = piece_emotion.unsqueeze(1).expand(-1, bar_predictions.size(1), -1)

        # Compute alignment loss
        # This ensures bar-level predictions align with piece-level emotions
        # Uses contrastive learning to make predictions similar to true emotions
        # and dissimilar to emotions from other pieces in the batch
        align_loss = self.compute_alignment_loss(bar_predictions, piece_emb_expanded)

        # Compute temporal regularization loss
        # This encourages smooth transitions between consecutive bars
        # by penalizing large emotional changes between adjacent bars
        temp_loss = self.compute_temporal_loss(bar_predictions)

        # Compute L2 regularization loss
        # Calculate L2 norm of predictions along emotion dimension
        # This prevents the model from making extreme predictions
        # by penalizing large values in the prediction vectors
        reg_loss = torch.mean(torch.norm(bar_predictions, p=2, dim=-1))

        # Combine all losses with their respective weights
        # - align_loss: Base loss for emotional alignment
        # - temp_loss: Weighted by lambda_reg (0.5) to control temporal smoothness
        # - reg_loss: Weighted by lambda_l2 (0.1) to control prediction magnitude
        total_loss = align_loss + self.lambda_reg * temp_loss + self.lambda_l2 * reg_loss

        # Return dictionary containing all loss components for logging and analysis
        return {
            "total_loss": total_loss,  # Combined weighted loss
            "align_loss": align_loss,  # Emotional alignment component
            "temp_loss": temp_loss,  # Temporal smoothness component
            "reg_loss": reg_loss,  # L2 regularization component
        }

    def compute_alignment_loss(self, bar_predictions, piece_emb_expanded, margin=0.1):
        """
        Compute contrastive alignment loss between bar predictions and piece emotions.

        Uses a margin-based ranking loss to ensure that:
        1. Bar predictions are similar to their corresponding piece emotions
        2. Bar predictions are dissimilar to emotions from other pieces in the batch

        Args:
            bar_predictions (torch.Tensor): Predicted emotions per bar [batch_size, seq_len, 5]
            piece_emb_expanded (torch.Tensor): Expanded piece emotions [batch_size, seq_len, 5]
            margin (float, optional): Margin for contrastive loss. Defaults to 0.1

        Returns:
            torch.Tensor: Alignment loss value
        """
        # Create cosine similarity function that computes similarity along the emotion dimension
        cos = nn.CosineSimilarity(dim=-1)  # Will output values between -1 and 1

        # Compute positive similarities
        # Calculate similarity between predicted emotions and their corresponding true emotions
        # Shape: [batch_size, seq_len] - one similarity score per bar
        pos_sim = cos(bar_predictions, piece_emb_expanded)

        # Generate negative samples for contrastive loss
        # Create negative pairs by shifting the piece emotions by one position in the batch
        # This effectively pairs each bar's predictions with emotions from a different piece
        piece_emb_neg = torch.roll(piece_emb_expanded, shifts=1, dims=0)  # Shape: [batch_size, seq_len, 5]

        # Calculate similarity between predictions and negative (mismatched) emotions
        neg_sim = cos(bar_predictions, piece_emb_neg)  # Shape: [batch_size, seq_len]

        # Compute margin-based ranking loss
        # The loss encourages:
        # - positive similarities (pos_sim) to be higher than negative similarities (neg_sim)
        # - by at least the margin amount
        # clamp ensures the loss is always non-negative
        loss = torch.mean(torch.clamp(self.margin - pos_sim + neg_sim, min=0))

        # Final loss interpretation:
        # - If pos_sim > neg_sim + margin: loss = 0 (good prediction)
        # - Otherwise: loss = margin - (pos_sim - neg_sim) (needs improvement)
        return loss

    def compute_temporal_loss(self, bar_predictions):
        """
        Compute temporal regularization loss to encourage smooth emotional transitions.

        The loss penalizes large changes in emotional content between consecutive bars,
        promoting more natural and gradual emotional progression.

        Args:
            bar_predictions (torch.Tensor): Predicted emotions per bar [batch_size, seq_len, 5]

        Returns:
            torch.Tensor: Temporal regularization loss value
        """
        # Set minimum similarity threshold between consecutive bars
        # Higher margin means we allow more emotional variation between bars
        margin = 0.4  # Threshold for acceptable emotional change between bars

        # Calculate similarity between consecutive bars
        # Using cosine similarity to measure how similar emotions are between adjacent bars
        consecutive_diffs = F.cosine_similarity(
            bar_predictions[:, 1:],  # Forward slice: all bars except the first one [batch_size, seq_len-1, 5]
            bar_predictions[:, :-1],  # Backward slice: all bars except the last one [batch_size, seq_len-1, 5]
            dim=-1,  # Compare along emotion dimension
        )  # Shape: [batch_size, seq_len-1] - similarity scores between consecutive bars

        # Compute temporal smoothness loss
        # 1. Subtract margin from similarities (consecutive_diffs - margin)
        # 2. Ensure numerical stability with clamp (prevent log(0))
        # 3. Take negative log to penalize low similarities more heavily
        # 4. Average across all consecutive bar pairs
        temp_loss = -torch.log(  # Negative log makes small similarities result in large loss
            torch.clamp(  # Ensure values don't get too close to zero
                consecutive_diffs - self.margin, min=1e-6  # How much the similarity exceeds our minimum threshold  # Numerical stability to prevent log(0)
            )
        ).mean()  # Average across all bar pairs

        # Higher loss indicates more abrupt emotional changes between consecutive bars
        # Lower loss indicates smoother emotional transitions
        return temp_loss

    def save_checkpoint(self, path, optimizer, epoch, loss):
        """
        Save model checkpoint with all necessary information for resuming training.

        Args:
            path (str): Path to save the checkpoint
            optimizer (torch.optim.Optimizer): The optimizer instance
            epoch (int): Current training epoch
            loss (float): Current loss value
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "d_model": self.d_model,
            "margin": self.margin,
            "lambda_reg": self.lambda_reg,
            "lambda_l2": self.lambda_l2,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path, feature_dim, d_model, eval=True, map_location=None):
        """
        Load a saved model checkpoint.

        Args:
            path (str): Path to the checkpoint file
            feature_dim (int): Dimension of input features
            d_model (int): Hidden dimension of the model
            eval (bool, optional): Whether to set model to evaluation mode. Defaults to True
            map_location (str, optional): Device to map model to. Defaults to None

        Returns:
            tuple: (EmotionMapper, dict) The loaded model and checkpoint data
        """
        checkpoint = torch.load(path, map_location=map_location)

        model = cls(feature_dim=feature_dim, d_model=d_model)
        model.load_state_dict(checkpoint["state_dict"])
        if eval:
            model.freeze()
            model.eval()
        return model, checkpoint

    @property
    def device(self):
        """Get the device the model parameters are on."""
        return next(self.parameters()).device

    def get_z(self, batch):
        """
        Extract input features from a batch dictionary.

        Handles various combinations of inputs:
        - Latents only
        - Symbolic Features only
        - Latents + Symbolic Features
        - Emotions only
        - Latents + Emotions
        - Symbolic Features + Emotions
        - Latents + Symbolic Features + Emotions

        Args:
            batch (dict): Batch dictionary containing various input features

        Returns:
            tuple: (dict, torch.Tensor) Input features dictionary and description bar IDs
        """
        z, desc_bar_ids = None, None
        if batch.get("latents") is not None and batch.get("description") is None and batch.get("emotions_vector") is None:
            z = {"latents": batch["latents"]}
        elif batch.get("description") is not None and batch.get("latents") is None and batch.get("emotions_vector") is None:
            z = {"description": batch["description"]}
            desc_bar_ids = batch["desc_bar_ids"]
        elif batch.get("latents") is not None and batch.get("description") is not None and batch.get("emotions_vector") is None:
            z = {"latents": batch["latents"], "description": batch["description"]}
            desc_bar_ids = batch["desc_bar_ids"]
        elif batch.get("emotions_vector") is not None and batch.get("latents") is None and batch.get("description") is None:
            z = {"emotions_vector": batch["emotions_vector"]}
        elif batch.get("latents") is not None and batch.get("emotions_vector") is not None and batch.get("description") is None:
            z = {"latents": batch["latents"], "emotions_vector": batch["emotions_vector"]}
        elif batch.get("description") is not None and batch.get("emotions_vector") is not None and batch.get("latents") is None:
            z = {"description": batch["description"], "emotions_vector": batch["emotions_vector"]}
            desc_bar_ids = batch["desc_bar_ids"]
        elif batch.get("latents") is not None and batch.get("description") is not None and batch.get("emotions_vector") is not None:
            z = {"latents": batch["latents"], "description": batch["description"], "emotions_vector": batch["emotions_vector"]}
            desc_bar_ids = batch["desc_bar_ids"]

        return z, desc_bar_ids

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (dict): Batch of training data
            batch_idx (int): Index of current batch

        Returns:
            torch.Tensor: Total loss value for the batch
        """
        z, desc_bar_ids = self.get_z(batch)

        # Get bar and position information
        bar_ids = batch.get("bar_ids")
        position_ids = batch.get("position_ids")

        # Forward pass
        output = self.forward(z, desc_bar_ids, bar_ids, position_ids)
        losses = self.compute_losses(output["predictions"], output["raw_emotions"], output["attention_weights"])

        # Log metrics
        self.log("train_loss", losses["total_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("align_loss", losses["align_loss"], on_step=True, on_epoch=True)
        self.log("temp_loss", losses["temp_loss"], on_step=True, on_epoch=True)

        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (dict): Batch of validation data
            batch_idx (int): Index of current batch

        Returns:
            torch.Tensor: Total loss value for the batch
        """
        z, desc_bar_ids = self.get_z(batch)

        # Forward pass
        output = self.forward(z, desc_bar_ids)
        losses = self.compute_losses(output["predictions"], output["raw_emotions"], output["attention_weights"])

        # Log validation metrics
        self.log("valid_loss", losses["total_loss"], on_step=True, on_epoch=True, prog_bar=True)

        return losses["total_loss"]

    def configure_optimizers(self):
        """Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer with learning rate 1e-4
        """
        optimizer = Adam(self.parameters(), lr=1e-4)
        return optimizer

    @torch.no_grad()
    def generate_bar_level_emotions_old(self, batch):
        """
        Generate bar-level emotion predictions for a batch of inputs.

        This method is used for inference and runs without gradient computation.
        It ensures predictions align with the actual number of bars in each piece.

        Args:
            batch (dict): Input batch containing features, emotions, and bar information

        Returns:
            dict: Dictionary containing:
                - predictions (torch.Tensor): Bar-level emotion predictions aligned with bars
                - raw_emotions (torch.Tensor): Input emotion vectors
                - attention_weights (torch.Tensor): Attention weights
                - emotion_names (list): List of emotion categories
                - avg_predictions (torch.Tensor): Average emotion predictions across all bars
        """
        # Get input features
        z, desc_bar_ids = self.get_z(batch)

        # Get bar and position information
        bar_ids = batch.get("bar_ids")
        position_ids = batch.get("position_ids")

        # Generate predictions with bar awareness
        output = self.forward(z, desc_bar_ids, bar_ids, position_ids)

        # Get the maximum number of bars
        max_bars = bar_ids.max().item() + 1

        # Ensure bar_ids matches predictions shape
        if bar_ids.size(1) != output["predictions"].size(1):
            # Truncate or pad bar_ids to match predictions sequence length
            if bar_ids.size(1) > output["predictions"].size(1):
                bar_ids = bar_ids[:, : output["predictions"].size(1)]
            else:
                # Pad with the last bar ID if bar_ids is shorter
                pad_size = output["predictions"].size(1) - bar_ids.size(1)
                last_ids = bar_ids[:, -1:].expand(-1, pad_size)
                bar_ids = torch.cat([bar_ids, last_ids], dim=1)

        # Reshape predictions to align with bars
        # Shape: [batch_size, max_bars, 5]
        predictions = torch.zeros(
            (output["predictions"].size(0), max_bars, output["predictions"].size(-1)), dtype=output["predictions"].dtype, device=output["predictions"].device
        )

        # Aggregate predictions for each bar
        for b in range(output["predictions"].size(0)):  # For each batch
            for bar in range(max_bars):
                # Get indices for current bar
                bar_mask = bar_ids[b] == bar
                if bar_mask.any():
                    # Average predictions for the current bar
                    predictions[b, bar] = output["predictions"][b, bar_mask].mean(dim=0)
                else:
                    # If no predictions for this bar, use the last valid prediction
                    # or the piece-level emotion if it's the first bar
                    if bar > 0:
                        predictions[b, bar] = predictions[b, bar - 1]
                    else:
                        predictions[b, bar] = output["raw_emotions"][b]

        # Calculate average predictions across all bars
        avg_predictions = predictions.mean(dim=1)

        return {
            "predictions": predictions,  # Now aligned with bars
            "raw_emotions": output["raw_emotions"],
            "attention_weights": output["attention_weights"],
            "avg_predictions": avg_predictions,
            "num_bars": max_bars,
        }

    def generate_bar_level_emotions(self, batch):
        """
        Generate bar-level emotion predictions for a batch of inputs.
        Similar to VAE's predict_step but focused on emotion mapping.

        Args:
            batch (dict): Batch dictionary containing:
                - input_ids: Token IDs for the input sequence
                - files: List of file paths (optional)
                - latents: Latent vectors (optional)
                - description: Symbolic features (optional)
                - emotions_vector: Piece-level emotions (optional)

        Returns:
            dict: Dictionary containing predictions for each piece:
                - predictions: Bar-level emotion predictions
                - attention_weights: Attention distribution
                - piece_emotions: Original piece-level emotions
        """
        predictions = {}
        events = batch["input_ids"]
        vocab = RemiVocab()

        for i, event_ids in enumerate(events):
            # Get the file name if available
            file = batch["files"][i] if "files" in batch else f"piece_{i}"

            # Extract bars from the events
            bars = get_bars(vocab.decode(event_ids))

            # print(bars, flush=True)

            # Create groups of events by bar
            groups = [event_ids[start:end] for start, end in zip(bars[:-1], bars[1:])]
            groups.append(event_ids[bars[-1] :])

            # Get BOS and EOS tokens
            bos, eos = get_bos_eos_events(vocab)
            bos = bos.to(self.device)
            eos = eos.to(self.device)

            # Process each bar
            bar_predictions = []
            attention_weights = []

            # Create input features dictionary
            z = {}
            if "latents" in batch:
                z["latents"] = batch["latents"][i].unsqueeze(0) if batch["latents"] is not None else None
            if "description" in batch:
                z["description"] = batch["description"][i].unsqueeze(0) if batch["description"] is not None else None
            if "emotions_vector" in batch:
                z["emotions_vector"] = batch["emotions_vector"][i].unsqueeze(0) if batch["emotions_vector"] is not None else None

            for bar in groups:
                # Prepare input sequence for the bar
                x = torch.cat([bos, bar, eos])[: self.context_size].unsqueeze(0)

                # Get bar and position IDs
                bar_ids = torch.zeros_like(x)
                position_ids = torch.arange(len(x), device=x.device).unsqueeze(0)

                # Generate predictions for the bar
                output = self.forward(z, desc_bar_ids=None, bar_ids=bar_ids, position_ids=position_ids)

                print("BAR", flush=True)
                print(bar, flush=True)
                print("OUTPUT", flush=True)
                print(output, flush=True)

                # Store predictions and attention weights
                bar_predictions.append(output["predictions"])
                attention_weights.append(output["attention_weights"])

            # Concatenate predictions for all bars
            piece_predictions = torch.cat(bar_predictions, dim=1)
            piece_attention = torch.cat(attention_weights, dim=2)

            avg_predictions = piece_predictions.mean(dim=1)

            # Store results for this piece
            predictions[file] = {
                "predictions": piece_predictions,
                "attention_weights": piece_attention,
                "piece_emotions": z.get("emotions_vector", None),
                "avg_predictions": avg_predictions,
            }

        return predictions
