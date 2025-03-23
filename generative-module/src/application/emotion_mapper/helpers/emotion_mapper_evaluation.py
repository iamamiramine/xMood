import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from typing import Dict, List, Tuple

def compute_emotion_conservation(bar_emotions: torch.Tensor, piece_emotions: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics to evaluate how well bar-level emotions preserve piece-level emotions.
    
    Args:
        bar_emotions: [batch_size, seq_len, num_emotions] tensor of bar-level emotion probabilities
        piece_emotions: [batch_size, num_emotions] tensor of piece-level emotion probabilities
        
    Returns:
        Dictionary containing:
            - kl_divergence: KL divergence between averaged bar emotions and piece emotions
            - cosine_similarity: Cosine similarity between averaged bar emotions and piece emotions
    """
    # Average bar-level emotions
    avg_bar_emotions = bar_emotions.mean(dim=1)  # [batch_size, num_emotions]
    
    # Compute KL divergence
    kl_div = F.kl_div(
        F.log_softmax(avg_bar_emotions, dim=-1),
        F.softmax(piece_emotions, dim=-1),
        reduction='batchmean'
    )
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(avg_bar_emotions, piece_emotions, dim=-1).mean()
    
    return {
        "kl_divergence": kl_div.item(),
        "cosine_similarity": cos_sim.item()
    }

def compute_temporal_coherence(bar_emotions: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics to evaluate the smoothness of emotional transitions.
    
    Args:
        bar_emotions: [batch_size, seq_len, num_emotions] tensor of bar-level emotion probabilities
        
    Returns:
        Dictionary containing:
            - avg_transition_rate: Average rate of emotion change between consecutive bars
            - max_transition_rate: Maximum rate of emotion change between consecutive bars
    """
    # Compute differences between consecutive bars
    emotion_diffs = torch.norm(
        bar_emotions[:, 1:] - bar_emotions[:, :-1],
        dim=-1
    )  # [batch_size, seq_len-1]
    
    avg_transition_rate = emotion_diffs.mean().item()
    max_transition_rate = emotion_diffs.max().item()
    
    return {
        "avg_transition_rate": avg_transition_rate,
        "max_transition_rate": max_transition_rate
    }

def analyze_attention_patterns(
    attention_weights: torch.Tensor,
    bar_emotions: torch.Tensor
) -> Dict[str, float]:
    """
    Analyze attention weight patterns and their correlation with emotional content.
    
    Args:
        attention_weights: [batch_size, num_heads, seq_len, seq_len] attention weights
        bar_emotions: [batch_size, seq_len, num_emotions] bar-level emotion probabilities
        
    Returns:
        Dictionary containing attention pattern metrics
    """
    # Average attention weights across heads
    avg_attention = attention_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
    
    # Compute attention entropy to measure focus/spread
    attention_entropy = -torch.sum(
        avg_attention * torch.log(avg_attention + 1e-10),
        dim=-1
    ).mean().item()
    
    # Compute attention concentration (how much attention focuses on key bars)
    attention_concentration = torch.max(avg_attention, dim=-1)[0].mean().item()
    
    return {
        "attention_entropy": attention_entropy,
        "attention_concentration": attention_concentration
    }

def evaluate_emotion_mapper(
    model_outputs: Dict[str, torch.Tensor],
    emotion_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive evaluation of emotion mapper outputs.
    
    Args:
        model_outputs: Dictionary containing:
            - predictions: [batch_size, seq_len, d_model] bar-level emotion embeddings
            - raw_emotions: [batch_size, num_emotions] piece-level emotion scores
            - attention_weights: [batch_size, num_heads, seq_len, seq_len] attention weights
            - emotion_values: [batch_size, seq_len, num_emotions] bar-level emotion probabilities
        emotion_names: List of emotion names
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    conservation_metrics = compute_emotion_conservation(
        model_outputs["emotion_values"],
        model_outputs["raw_emotions"]
    )
    
    temporal_metrics = compute_temporal_coherence(
        model_outputs["emotion_values"]
    )
    
    attention_metrics = analyze_attention_patterns(
        model_outputs["attention_weights"],
        model_outputs["emotion_values"]
    )
    
    return {
        "conservation_metrics": conservation_metrics,
        "temporal_metrics": temporal_metrics,
        "attention_metrics": attention_metrics
    }

def plot_emotion_heatmap(
    emotion_values: torch.Tensor,
    emotion_names: List[str],
    save_path: str = None
) -> None:
    """
    Create a heatmap visualization of emotions across bars.
    
    Args:
        emotion_values: [seq_len, num_emotions] tensor of bar-level emotion probabilities
        emotion_names: List of emotion names
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert to numpy for plotting
    emotion_values_np = emotion_values.cpu().numpy()
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        emotion_values_np.T,
        yticklabels=emotion_names,
        xticklabels=list(range(emotion_values_np.shape[0])),
        cmap='viridis',
        cbar_kws={'label': 'Emotion Intensity'}
    )
    
    plt.xlabel('Bar Number')
    plt.ylabel('Emotion')
    plt.title('Emotion Distribution Across Bars')
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 