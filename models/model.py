import torch
import torch.nn as nn

from models.encoders.text_encoder import TextEncoder
from models.encoders.image_encoder import ImageEncoder
from models.fusion.projection import Projection
from models.fusion.cross_attention import CrossAttention
from models.fusion.gating import Gating

class CLIPCACG(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()

        self.text_proj = Projection(768)
        self.image_proj = Projection(2048)

        self.cross_attn = CrossAttention()
        self.gate = Gating()

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask, images, image_mask):
        # -----------------
        # 1. TEXT PROCESSING
        # -----------------
        t = self.text_encoder(input_ids, attention_mask)
        t = self.text_proj(t)

       # ------------------
        # 2. IMAGE PROCESSING
        # ------------------
        B, N, C, H, W = images.shape
        
        # Flatten and extract visual features
        i = self.image_encoder(images.view(B * N, C, H, W))
        i = i.view(B, N, -1)  # Shape: (B, N, 2048)

        # The dataloader now supplies a perfect binary mask matching the tensor layout
        mask = image_mask.to(i.device)  # Shape: (B, N, 1)

        # Apply mask and compute the true average
        i = i * mask 
        i = i.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # Shape: (B, 2048)
        i = self.image_proj(i)

        # -----------------------
        # 3. CROSS ATTENTION & FUSION
        # -----------------------
        # Note: If your CrossAttention layer expects images to retain their sequence 
        # dimension (B, N, D), move this step BEFORE the `.sum(dim=1)` pooling above!
        t, i = self.cross_attn(t, i)

        # GATING & CLASSIFIER
        f = self.gate(t, i)
        out = self.classifier(f)

        return out