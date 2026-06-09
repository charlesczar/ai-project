import torch.nn as nn
from transformers import RobertaModel

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RobertaModel.from_pretrained("roberta-base")

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output  # (B, 768)