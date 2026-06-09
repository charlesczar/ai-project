import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim=512, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, text, image):
        t, _ = self.attn(text.unsqueeze(1), image.unsqueeze(1), image.unsqueeze(1))
        i, _ = self.attn(image.unsqueeze(1), text.unsqueeze(1), text.unsqueeze(1))

        return t.squeeze(1), i.squeeze(1)