import yaml
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from torchvision import transforms

from datasets.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
from models.model import CLIPCACG
from training.eval import evaluate

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the valuation CSV file")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint (.pt file)")
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
    dataset = MultiModalDataset(df, tokenizer, transform, img_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=multimodal_collate_fn
    )

    model = CLIPCACG(num_classes=3).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    evaluate(model, dataloader, device)
