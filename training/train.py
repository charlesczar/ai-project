import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from torchvision import transforms
from sklearn.model_selection import train_test_split

from datasets.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
from models.model import CLIPCACG
from training.engine import Trainer
from training.eval import evaluate

def main(df, config, base_dir):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = RobertaTokenizer.from_pretrained(config["model"]["text_model"])

    image_size = config["data"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    train_df, val_df = train_test_split(
        df,
        test_size=config["data"]["val_split"],
        random_state=42,
        stratify=df["label"]
    )

    img_dir = str(base_dir / "data" / "images")

    train_dataset = MultiModalDataset(train_df.reset_index(drop=True), tokenizer, transform, img_dir)
    val_dataset = MultiModalDataset(val_df.reset_index(drop=True), tokenizer, transform, img_dir)

    loader_kwargs = dict(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        collate_fn=multimodal_collate_fn
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = CLIPCACG(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    trainer = Trainer(model, optimizer, device)

    checkpoint_dir = base_dir / config["output"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        loss = trainer.train_one_epoch(train_loader)
        print(f"Epoch {epoch+1}/{config['training']['epochs']}  Train Loss: {loss:.4f}")
        evaluate(model, val_loader, device)
        torch.save(model.state_dict(), checkpoint_dir / f"model_epoch_{epoch}.pt")