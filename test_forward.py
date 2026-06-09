import torch
import pandas as pd
from transformers import RobertaTokenizer
from torchvision import transforms

from datasets.multimodal_dataset import MultiModalDataset
from models.model import CLIPCACG

# -------------------------
# LOAD CSV
# -------------------------
df = pd.read_csv("data/reviews.csv")

# -------------------------
# TOKENIZER
# -------------------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# -------------------------
# IMAGE TRANSFORM
# -------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------
# DATASET
# -------------------------
dataset = MultiModalDataset(
    df=df,
    tokenizer=tokenizer,
    image_transform=image_transform,
    img_dir="data/images"
)

# -------------------------
# MODEL
# -------------------------
model = CLIPCACG(num_classes=3)
model.eval()

# -------------------------
# GET ONE SAMPLE
# -------------------------
sample = dataset[0]

input_ids = sample["input_ids"].unsqueeze(0)
attention_mask = sample["attention_mask"].unsqueeze(0)
images = sample["images"].unsqueeze(0)

# -------------------------
# FORWARD PASS TEST
# -------------------------
with torch.no_grad():
    output = model(input_ids, attention_mask, images)

print("Output shape:", output.shape)
print("Output logits:", output)