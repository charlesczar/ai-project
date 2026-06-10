import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim=512, heads=4):
        super().__init__()
        # batch_first=True expects shapes: [Batch, Sequence, Features]
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, text, image):
        # 1. Dynamically ensure text is a 3D sequence tensor [B, S_text, D]
        if text.dim() == 2:
            text = text.unsqueeze(1)
            
        # 2. Dynamically ensure image is a 3D sequence tensor [B, S_image, D]
        if image.dim() == 2:
            image = image.unsqueeze(1)

        # 3. Co-Attention Math (Query, Key, Value)
        # Text attending to Image
        t_out, _ = self.attn(query=text, key=image, value=image)
        
        # Image attending to Text
        i_out, _ = self.attn(query=image, key=text, value=text)

        # 4. Safely return to 2D features if they only had 1 sequence token
        # If they had multiple sequence tokens (like multiple images), we keep the tokens!
        if t_out.shape[1] == 1:
            t_out = t_out.squeeze(1)
        if i_out.shape[1] == 1:
            i_out = i_out.squeeze(1)

        return t_out, i_out