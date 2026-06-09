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

    def forward(self, input_ids, attention_mask, images):

        # TEXT
        t = self.text_encoder(input_ids, attention_mask)
        t = self.text_proj(t)

        # IMAGE (multiple images → average)
        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W)

        i = self.image_encoder(images)
        i = i.view(B, N, -1).mean(dim=1)  # average pooling
        i = self.image_proj(i)

        # CROSS ATTENTION
        t, i = self.cross_attn(t, i)

        # GATING
        f = self.gate(t, i)

        # CLASSIFIER
        out = self.classifier(f)

        return out