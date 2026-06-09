import argparse
from pathlib import Path
import re
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from torchvision import transforms

from datasets.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
from models.model import CLIPCACG


# -------------------------
# GET LATEST CHECKPOINT
# -------------------------
def get_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    files = list(checkpoint_dir.glob("model_epoch_*.pt"))

    if not files:
        raise FileNotFoundError("No checkpoints found")

    def extract_epoch(path):
        match = re.findall(r"\d+", path.stem)
        return int(match[-1]) if match else -1

    return max(files, key=extract_epoch)


# -------------------------
# MAIN
# -------------------------
def main(checkpoint_path, csv_path, base_dir, config_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # config
    config = yaml.safe_load(Path(config_path).read_text())

    # data
    df = pd.read_csv(csv_path)

    # tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(
        config["model"]["text_model"]
    )

    # image transform
    image_size = config["data"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    img_dir = str(Path(base_dir) / "data" / "images")

    dataset = MultiModalDataset(
        df.reset_index(drop=True),
        tokenizer,
        transform,
        img_dir
    )

    loader = DataLoader(
        dataset,
        batch_size=1,  # IMPORTANT: for clean readable output
        shuffle=False,
        collate_fn=multimodal_collate_fn
    )

    # model
    model = CLIPCACG(
        num_classes=config.get("model", {}).get("num_classes", 3)
    ).to(device)

    # load checkpoint
    if checkpoint_path == "latest":
        checkpoint_path = get_latest_checkpoint("checkpoints/")
        print(f"[INFO] Loading latest checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(
        checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    )

    model.eval()

    # -------------------------
    # PREDICTION LOOP
    # -------------------------
    class_names = ["Negative", "Neutral", "Positive"]

    print("\n===== PREDICTIONS =====\n")

    for i, batch in enumerate(loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["images"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            # compute or use provided image_mask
            if "image_mask" in batch:
                image_mask = batch["image_mask"].to(device)
            else:
                image_mask = torch.ones(input_ids.size(0), images.size(1), device=device, dtype=torch.float)

            outputs = model(input_ids, attention_mask, images, image_mask)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

        print(f"Sample {i+1}")
        print(f"True Label: {class_names[labels.item()]}")
        print(f"Predicted : {class_names[preds.item()]}")
        print(f"Confidence: {probs[0][preds].item():.4f}")
        print("-" * 40)


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    main(args.checkpoint, args.csv, args.base_dir, args.config)