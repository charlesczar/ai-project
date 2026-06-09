import yaml
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer
from torchvision import transforms
from PIL import Image

from models.model import CLIPCACG
from datasets.multimodal_dataset import multimodal_collate_fn

BASE_DIR = Path(__file__).resolve().parent

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


class PredictionDataset(Dataset):
    """Like MultiModalDataset but does not require a label column."""

    def __init__(self, df, tokenizer, image_transform, img_dir):
        self.df = df
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def _load_images(self, image_list):
        images = []
        for img_name in image_list:
            path = f"{self.img_dir}/{img_name}"
            img = Image.open(path).convert("RGB")
            images.append(self.image_transform(img))
        return torch.stack(images)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = (
            str(row["product_title"]) + " " +
            str(row["product_description"]) + " " +
            str(row["review_text"])
        )
        text_enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        image_files = [f.strip().strip('"') for f in str(row["image_paths"]).split(";")]

        return {
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "images": self._load_images(image_files),
            "label": torch.tensor(-1, dtype=torch.long),  # placeholder — not used
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to input CSV (no label column required)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--output", default="predictions.csv", help="Where to save the output CSV")
    args = parser.parse_args()

    with open(BASE_DIR / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(args.csv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = RobertaTokenizer.from_pretrained(config["model"]["text_model"])
    image_size = config["data"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    img_dir = str(BASE_DIR / "data" / "images")
    dataset = PredictionDataset(df, tokenizer, transform, img_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=multimodal_collate_fn
    )

    model = CLIPCACG(num_classes=3).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["images"].to(device)
            image_mask = batch["image_mask"].to(device)

            logits = model(input_ids, attention_mask, images, image_mask)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    df["predicted_label"] = all_preds
    df["predicted_sentiment"] = [LABEL_MAP[p] for p in all_preds]
    df["prob_negative"] = [round(float(p[0]), 4) for p in all_probs]
    df["prob_neutral"] = [round(float(p[1]), 4) for p in all_probs]
    df["prob_positive"] = [round(float(p[2]), 4) for p in all_probs]

    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(df[["product_title", "predicted_sentiment", "prob_negative", "prob_neutral", "prob_positive"]].to_string())
