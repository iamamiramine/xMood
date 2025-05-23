import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """
    Embedding fusion module that combines embeddings from multiple modalities 
    using cross-attention mechanism similar to EmoMusicTV.
    """
    
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Multi-head attention for cross-modality attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query_embed, key_value_embeds):
        """
        Forward pass of the cross-attention fusion module.
        
        Args:
            query_embed: Tensor of shape (batch_size, embed_dim) - main embedding
            key_value_embeds: List of tensors, each of shape (batch_size, embed_dim)
                             - supporting embeddings
            
        Returns:
            Tensor of shape (batch_size, embed_dim) - fused embedding
        """
        # Convert tensors to expected shape for MultiheadAttention
        # MultiheadAttention expects (seq_len, batch_size, embed_dim)
        query = query_embed.unsqueeze(0)  # (1, batch_size, embed_dim)
        
        # Concatenate all key-value embeddings
        # Resulting shape: (num_embeds, batch_size, embed_dim)
        keys_values = torch.cat([embed.unsqueeze(0) for embed in key_value_embeds], dim=0)
        
        # Apply cross-attention
        attn_output, _ = self.cross_attention(
            query=query,
            key=keys_values,
            value=keys_values
        )
        
        # Apply residual connection and normalization
        attn_output = query + attn_output
        attn_output = self.norm1(attn_output)
        
        # Apply feed-forward network with residual connection
        ffn_output = self.ffn(attn_output)
        ffn_output = attn_output + ffn_output
        ffn_output = self.norm2(ffn_output)
        
        # Remove sequence dimension and return
        return ffn_output.squeeze(0)


class LinearConcatFusion(nn.Module):
    """
    Alternative embedding fusion module that combines embeddings from multiple modalities 
    using concatenation and linear projection.
    """
    
    def __init__(self, input_dims, output_dim=512, dropout=0.1):
        super(LinearConcatFusion, self).__init__()
        
        # Calculate total input dimension
        total_input_dim = sum(input_dims)
        
        # Create linear projection layer
        self.projection = nn.Sequential(
            nn.Linear(total_input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, *embeddings):
        """
        Forward pass of the linear concatenation fusion module.
        
        Args:
            *embeddings: Variable number of embedding tensors,
                        each of shape (batch_size, embed_dim)
            
        Returns:
            Tensor of shape (batch_size, output_dim) - fused embedding
        """
        # Concatenate all embeddings along the embedding dimension
        concat_embed = torch.cat(embeddings, dim=1)
        
        # Project to the desired output dimension
        return self.projection(concat_embed) 