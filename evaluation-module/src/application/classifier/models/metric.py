import torch.nn as nn
import numpy as np
import torch

def accuracy(source, target):
    # Handle mood vector case - compare highest probability mood
    if len(target.shape) > 1 and target.shape[1] > 1:
        source_idx = source.max(1)[1].long()
        target_idx = target.max(1)[1].long()
        correct = (source_idx == target_idx).sum()
        mean_correct = correct / source.shape[0]
        return mean_correct
    # Original code for valence-arousal
    else:
        source = source.max(1)[1].long()
        target = target.long()
        correct = (source == target).sum()
        mean_correct = correct / source.shape[0]
        return mean_correct

def kl_divergence(source, target):
    """
    Calculates KL divergence between predicted and target probability distributions
    for mood vector evaluation.
    """
    # Add small epsilon to avoid log(0)
    source = torch.clamp(source, min=1e-8, max=1.0)
    target = torch.clamp(target, min=1e-8, max=1.0)
    
    # Calculate KL divergence: sum(target * log(target/source))
    kl_div = torch.sum(target * torch.log(target/source), dim=1)
    mean_kl_div = kl_div.mean()
    return mean_kl_div