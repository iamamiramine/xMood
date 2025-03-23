"""
This module implements an emotion mapping model that translates symbolic features into bar-level emotion predictions.
The model uses a transformer-based architecture to map textual symbolic features to a sequence of emotion vectors.
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

# Local application imports
from application.encoder.models.vocab_model import SymbolicFeaturesVocab, RemiVocab, MoodsVocab
from domain.constants.encoder.token_constants import BAR_KEY
from application.emotion_mapper.models.emotion_attention import ScaledDotProductAttention


class EmotionMapper(pl.LightningModule):
    """
    A PyTorch Lightning module that maps symbolic features to bar-level emotion predictions.

    The model uses a transformer-based architecture with:
    - Symbolic feature embeddings (based on SymbolicFeaturesVocab)
    - Emotion embeddings (5 emotions)
    - Multi-head self-attention to learn relationships
    - Emotion projection and decoding layers

    Args:
        d_model (int, optional): Hidden dimension of the model. Defaults to 512
        num_heads (int, optional): Number of attention heads. Defaults to 8
        symbolic_dim (int, optional): Dimension of symbolic features. Defaults to None
    """

    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model

        # Initialize vocabularies
        self.symb_vocab = SymbolicFeaturesVocab()
        self.emotion_vocab = MoodsVocab()

        symbolic_dim = len(self.symb_vocab)
        emotion_dim = len(self.emotion_vocab)

        # Input projections for symbolic features and emotions
        self.symbolic_projector = nn.Sequential(nn.Linear(symbolic_dim, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.emotion_projector = nn.Sequential(nn.Linear(5, d_model), nn.LayerNorm(d_model), nn.ReLU())

        # Replace MultiheadAttention with ScaledDotProductAttention
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_model, d_v=d_model, h=num_heads)
        self.norm = nn.LayerNorm(d_model)

        # Emotion decoder - maps from model dimension back to 5 emotions
        self.emotion_decoder = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 5), nn.Softmax(dim=-1))

        # Loss function hyperparameters
        self.lambda_reg = 0.5
        self.lambda_l2 = 0.1

    def encode_features(self, symbolic_features):
        """
        Projects symbolic features into model dimension space.

        Args:
            symbolic_features (list or torch.Tensor): Symbolic features as list of lists or tensor

        Returns:
            torch.Tensor: Encoded features [batch_size, seq_len, d_model]
        """
        if symbolic_features is None:
            return None

        # Convert list of lists to tensor if needed
        if isinstance(symbolic_features, list):
            # Find the maximum length among all feature vectors
            max_len = max(len(features) for features in symbolic_features)

            # Pad each feature vector to max_len with zeros
            padded_features = []
            for features in symbolic_features:
                if len(features) < max_len:
                    # Create a padded version of the features
                    padded = torch.zeros(max_len, dtype=torch.float32)
                    padded[: len(features)] = torch.tensor(features, dtype=torch.float32)
                    padded_features.append(padded)
                else:
                    padded_features.append(torch.tensor(features, dtype=torch.float32))

            symbolic_features = torch.stack(padded_features)

        # Ensure tensor is on the correct device
        symbolic_features = symbolic_features.to(self.device)

        return self.symbolic_projector(symbolic_features)

    def encode_emotions(self, emotions_vector):
        """
        Projects 5-dimensional emotion vectors into model dimension space.

        Args:
            emotions_vector (torch.Tensor): Emotion scores [batch_size, 5]

        Returns:
            torch.Tensor: Encoded emotions [batch_size, d_model]
        """
        if emotions_vector is None:
            return None

        return self.emotion_projector(emotions_vector)

    def forward(self, symbolic_features, emotions_vector=None):
        """
        Forward pass of the emotion mapper model.

        Args:
            symbolic_features (torch.Tensor): Symbolic features [batch_size, seq_len, symbolic_dim]
            emotions_vector (torch.Tensor, optional): Emotion vector [batch_size, 5]

        Returns:
            dict: Dictionary containing:
                - predictions (torch.Tensor): Bar-level emotion predictions [batch_size, seq_len, 5]
                - raw_emotions (torch.Tensor): Input emotion vectors [batch_size, 5]
                - attention_weights (torch.Tensor): Attention weights
        """
        # Project inputs to same dimension
        symbolic_emb = self.encode_features(symbolic_features)  # [batch_size, seq_len, d_model]
        emotion_emb = self.encode_emotions(emotions_vector)  # [batch_size, d_model]

        if symbolic_emb is None or emotion_emb is None:
            raise ValueError("Both symbolic features and emotion vector are required")

        # Apply attention between bars and emotion
        attended = self.attention(
            queries=emotion_emb.unsqueeze(1),  # [batch_size, seq_len, d_model]
            keys=symbolic_emb,  # [batch_size, 1, d_model]
            values=symbolic_emb,  # [batch_size, 1, d_model]
        )  # [batch_size, seq_len, d_model]

        # Add residual connection and normalize
        combined = self.norm(attended + symbolic_emb)

        # Generate predictions
        predictions = self.emotion_decoder(combined)  # [batch_size, seq_len, 5]

        return {
            "predictions": predictions,
            "raw_emotions": emotions_vector,
            "attention_weights": attended,  # Get attention weights from the attention module
        }

    def compute_losses(self, bar_predictions, piece_emotion, attention_weights):
        """
        Compute the combined loss for training the emotion mapper.

        The loss consists of three components:
        1. Alignment loss: Ensures predictions match the piece-level emotions
        3. L2 regularization: Prevents predictions from growing too large

        Args:
            bar_predictions (torch.Tensor): Predicted emotions per bar [batch_size, seq_len, 5]
            piece_emotion (torch.Tensor): Ground truth piece-level emotions [batch_size, 5]
            attention_weights (torch.Tensor): Attention weights [batch_size, num_heads, seq_len, seq_len]

        Returns:
            dict: Dictionary containing:
                - total_loss (torch.Tensor): Combined loss value
                - align_loss (torch.Tensor): Alignment loss component
                - reg_loss (torch.Tensor): L2 regularization loss
        """
        # Expand piece emotions to match the shape of bar predictions by:
        # 1. Adding sequence dimension (unsqueeze)
        # 2. Repeating emotions for each bar position (expand)
        # Result shape: [batch_size, seq_len, 5]c
        piece_emb_expanded = piece_emotion.unsqueeze(1).expand(-1, bar_predictions.size(1), -1)

        # Compute alignment loss
        # This ensures bar-level predictions align with piece-level emotions
        # Uses contrastive learning to make predictions similar to true emotions
        # and dissimilar to emotions from other pieces in the batch
        align_loss = self.compute_alignment_loss(bar_predictions, piece_emb_expanded)

        # Compute L2 regularization loss
        # Calculate L2 norm of predictions along emotion dimension
        # This prevents the model from making extreme predictions
        # by penalizing large values in the prediction vectors
        reg_loss = torch.mean(torch.norm(bar_predictions, p=2, dim=-1))

        # Combine all losses with their respective weights
        # - align_loss: Base loss for emotional alignment
        # - temp_loss: Weighted by lambda_reg (0.5) to control temporal smoothness
        # - reg_loss: Weighted by lambda_l2 (0.1) to control prediction magnitude
        total_loss = align_loss + self.lambda_l2 * reg_loss  # + self.lambda_reg * temp_loss

        # Return dictionary containing all loss components for logging and analysis
        return {
            "total_loss": total_loss,  # Combined weighted loss
            "align_loss": align_loss,  # Emotional alignment component
            "reg_loss": reg_loss,  # L2 regularization component
        }

    def compute_alignment_loss(self, bar_predictions, piece_emb_expanded, alpha=2):
        """
        Compute L2 alignment loss between each bar prediction and piece emotions.

        Args:
            bar_predictions (torch.Tensor): Predicted emotions per bar [batch_size, seq_len, 5]
            piece_emb_expanded (torch.Tensor): Expanded piece emotions [batch_size, seq_len, 5]
            alpha (float, optional): Power for the L2 norm. Defaults to 2

        Returns:
            torch.Tensor: Average alignment loss across all bars
        """
        # Reshape tensors to align each bar prediction with its target
        # From [batch_size, seq_len, 5] to [batch_size * seq_len, 5]
        bar_predictions_flat = bar_predictions.reshape(-1, bar_predictions.size(-1))
        piece_emb_flat = piece_emb_expanded.reshape(-1, piece_emb_expanded.size(-1))

        # Compute L2 loss between predictions and targets
        loss = (bar_predictions_flat - piece_emb_flat).norm(p=2, dim=1).pow(alpha).mean()

        return loss

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
    def load_checkpoint(cls, path, d_model, eval=True, map_location=None):
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

        model = cls(d_model=d_model)
        model.load_state_dict(checkpoint["state_dict"])
        if eval:
            model.freeze()
            model.eval()
        return model, checkpoint

    @property
    def device(self):
        """Get the device the model parameters are on."""
        return next(self.parameters()).device

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (dict): Batch of training data
            batch_idx (int): Index of current batch

        Returns:
            torch.Tensor: Total loss value for the batch
        """
        batch_symbolic = []
        for idx, file in enumerate(batch["symbolic"]):
            symbolic = batch["symbolic"][idx]
            bars = [i for i, event in enumerate(symbolic) if f"{BAR_KEY}_" in event]
            symbolic_groups = [symbolic[start:end] for start, end in zip(bars[:-1], bars[1:])]

            piece_symbolic = []
            for i, bar_symbolic_features in enumerate(symbolic_groups):
                # Create one-hot encoded tensor for each event in the bar
                vocab_size = len(self.symb_vocab)
                event_ids = self.symb_vocab.encode(bar_symbolic_features[1:])

                # Convert to one-hot encoding
                one_hot = torch.zeros(len(event_ids), vocab_size, dtype=torch.float32)
                for j, event_id in enumerate(event_ids):
                    one_hot[j, event_id] = 1.0

                # Average the one-hot vectors to get a single feature vector for the bar
                bar_features = one_hot.mean(dim=0)  # Shape: [vocab_size]
                piece_symbolic.append(bar_features)

            # Stack all bar features for this piece
            if piece_symbolic:
                piece_symbolic = torch.stack(piece_symbolic)  # Shape: [num_bars, vocab_size]
                batch_symbolic.append(piece_symbolic)

        if not batch_symbolic:
            raise ValueError("No valid symbolic features found in batch")

        # Pad sequences to same length and stack into batch
        max_bars = max(piece.size(0) for piece in batch_symbolic)
        padded_symbolic = []
        for piece in batch_symbolic:
            if piece.size(0) < max_bars:
                padding = torch.zeros(max_bars - piece.size(0), piece.size(1), dtype=torch.float32)
                padded_piece = torch.cat([piece, padding], dim=0)
            else:
                padded_piece = piece
            padded_symbolic.append(padded_piece)

        batch_symbolic = torch.stack(padded_symbolic)  # Shape: [batch_size, max_bars, vocab_size]
        batch_symbolic = batch_symbolic.to(self.device)

        output = self.forward(batch_symbolic, batch["emotions_vector"])
        losses = self.compute_losses(output["predictions"], output["raw_emotions"], output["attention_weights"])

        # Log metrics
        self.log("train_loss", losses["total_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("align_loss", losses["align_loss"], on_step=True, on_epoch=True)
        # self.log("temp_loss", losses["temp_loss"], on_step=True, on_epoch=True)

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
        batch_symbolic = []
        for idx, file in enumerate(batch["symbolic"]):
            symbolic = batch["symbolic"][idx]
            bars = [i for i, event in enumerate(symbolic) if f"{BAR_KEY}_" in event]
            symbolic_groups = [symbolic[start:end] for start, end in zip(bars[:-1], bars[1:])]

            piece_symbolic = []
            for i, bar_symbolic_features in enumerate(symbolic_groups):
                # Create one-hot encoded tensor for each event in the bar
                vocab_size = len(self.symb_vocab)
                event_ids = self.symb_vocab.encode(bar_symbolic_features[1:])

                # Convert to one-hot encoding
                one_hot = torch.zeros(len(event_ids), vocab_size, dtype=torch.float32)
                for j, event_id in enumerate(event_ids):
                    one_hot[j, event_id] = 1.0

                # Average the one-hot vectors to get a single feature vector for the bar
                bar_features = one_hot.mean(dim=0)  # Shape: [vocab_size]
                piece_symbolic.append(bar_features)

            # Stack all bar features for this piece
            if piece_symbolic:
                piece_symbolic = torch.stack(piece_symbolic)  # Shape: [num_bars, vocab_size]
                batch_symbolic.append(piece_symbolic)

        if not batch_symbolic:
            raise ValueError("No valid symbolic features found in batch")

        # Pad sequences to same length and stack into batch
        max_bars = max(piece.size(0) for piece in batch_symbolic)
        padded_symbolic = []
        for piece in batch_symbolic:
            if piece.size(0) < max_bars:
                padding = torch.zeros(max_bars - piece.size(0), piece.size(1), dtype=torch.float32)
                padded_piece = torch.cat([piece, padding], dim=0)
            else:
                padded_piece = piece
            padded_symbolic.append(padded_piece)

        batch_symbolic = torch.stack(padded_symbolic)  # Shape: [batch_size, max_bars, vocab_size]
        batch_symbolic = batch_symbolic.to(self.device)

        output = self.forward(batch_symbolic, batch["emotions_vector"])
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

    def generate_bar_level_emotions(self, batch):
        """
        Generate bar-level emotion predictions for a batch of inputs.
        Similar to VAE's predict_step but focused on emotion mapping.

        Args:
            batch (dict): Batch dictionary containing:
                - symbolic_raw: List of symbolic sequences
                - emotions_vector: Piece-level emotions (optional)

        Returns:
            dict: Dictionary containing predictions for each piece:
                - predictions: Bar-level emotion predictions [num_bars, 5]
                - attention_weights: Attention distribution
                - emotions_vector: Original piece-level emotions
                - avg_predictions: Average emotions across all bars [5]
        """
        batch_symbolic = []
        for idx, file in enumerate(batch["symbolic"]):
            symbolic = batch["symbolic"][idx]
            bars = [i for i, event in enumerate(symbolic) if f"{BAR_KEY}_" in event]
            symbolic_groups = [symbolic[start:end] for start, end in zip(bars[:-1], bars[1:])]

            piece_symbolic = []
            for i, bar_symbolic_features in enumerate(symbolic_groups):
                # Create one-hot encoded tensor for each event in the bar
                vocab_size = len(self.symb_vocab)
                event_ids = self.symb_vocab.encode(bar_symbolic_features[1:])

                # Convert to one-hot encoding
                one_hot = torch.zeros(len(event_ids), vocab_size, dtype=torch.float32)
                for j, event_id in enumerate(event_ids):
                    one_hot[j, event_id] = 1.0

                # Average the one-hot vectors to get a single feature vector for the bar
                bar_features = one_hot.mean(dim=0)  # Shape: [vocab_size]
                piece_symbolic.append(bar_features)

            # Stack all bar features for this piece
            if piece_symbolic:
                piece_symbolic = torch.stack(piece_symbolic)  # Shape: [num_bars, vocab_size]
                batch_symbolic.append(piece_symbolic)

        if not batch_symbolic:
            raise ValueError("No valid symbolic features found in batch")

        # Pad sequences to same length and stack into batch
        max_bars = max(piece.size(0) for piece in batch_symbolic)
        padded_symbolic = []
        for piece in batch_symbolic:
            if piece.size(0) < max_bars:
                padding = torch.zeros(max_bars - piece.size(0), piece.size(1), dtype=torch.float32)
                padded_piece = torch.cat([piece, padding], dim=0)
            else:
                padded_piece = piece
            padded_symbolic.append(padded_piece)

        batch_symbolic = torch.stack(padded_symbolic)  # Shape: [batch_size, max_bars, vocab_size]
        batch_symbolic = batch_symbolic.to(self.device)

        # Get model predictions
        with torch.no_grad():
            output = self.forward(batch_symbolic, batch.get("emotions_vector"))

        # Process predictions for each piece
        predictions = {}
        for idx, file in enumerate(batch["files"]):
            # Get number of actual bars for this piece (before padding)
            num_bars = len([e for e in batch["symbolic"][idx] if f"{BAR_KEY}_" in e]) - 1

            # Extract predictions for actual bars (remove padding)
            piece_predictions = output["predictions"][idx, :num_bars]  # [num_bars, 5]
            piece_attention = output["attention_weights"][idx, :, :num_bars]  # [num_heads, num_bars]

            # Calculate average predictions across bars
            avg_predictions = piece_predictions.mean(dim=0)  # [5]

            # Store results for this piece
            predictions[str(file)] = {
                "predictions": piece_predictions.cpu(),
                "attention_weights": piece_attention.cpu(),
                "emotions_vector": batch.get("emotions_vector")[idx].cpu() if batch.get("emotions_vector") is not None else None,
                "avg_predictions": avg_predictions.cpu(),
            }

        return predictions
