import torch
import torch.nn as nn

class Gating(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, text, image):
        g = torch.sigmoid(self.fc(torch.cat([text, image], dim=1)))
        return g * text + (1 - g) * image