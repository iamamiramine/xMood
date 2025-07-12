import torch
import torch.nn as nn
import clip

class ImageEncoder(nn.Module):
    """
    Image encoder using CLIP's Vision Transformer to process image inputs.
    It can be used with a frozen CLIP model or allow for fine-tuning.
    """
    
    def __init__(self, output_dim=384, pretrained=True, freeze_clip=True):
        super(ImageEncoder, self).__init__()
        
        # Load CLIP model's visual part
        # Note: requires `pip install git+https://github.com/openai/CLIP.git`
        clip_model, _ = clip.load('ViT-B/32', device='cpu')  # Load to cpu, will be moved by Lightning
        self.visual = clip_model.visual
        
        # If not using pretrained weights, re-initialize
        if not pretrained:
            self.visual.apply(self._weights_init)

        # Freeze CLIP model parameters if specified
        if freeze_clip:
            for param in self.visual.parameters():
                param.requires_grad = False
        
        # Projection layer to get the desired output dimension
        self.projection = nn.Linear(self.visual.output_dim, output_dim)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, images):
        """
        Forward pass of the image encoder.
        
        Args:
            images: Tensor of preprocessed images from the CLIP processor.
                    Expected shape: (batch_size, channels, height, width)
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        # Get CLIP image embeddings
        # The visual model is either frozen or will be trained, so we don't need a no_grad context here
        image_features = self.visual(images).to(images.dtype)
        
        # Project to the desired output dimension
        return self.projection(image_features) 