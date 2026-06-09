import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from torchvision import transforms

from datasets.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
from models.model import CLIPCACG
from training.engine import Trainer

def main(df, config, base_dir):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = RobertaTokenizer.from_pretrained(config["model"]["text_model"])

    image_size = config["data"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    img_dir = str(base_dir / "data" / "images")
    dataset = MultiModalDataset(df, tokenizer, transform, img_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=multimodal_collate_fn
    )

    print("Train dataset rows (dataset.len):", len(dataset))
    print("Train batches:", len(dataloader), " batch_size:", config["training"]["batch_size"])

    model = CLIPCACG(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    trainer = Trainer(model, optimizer, device)

    checkpoint_dir = base_dir / config["output"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        loss = trainer.train_one_epoch(dataloader)
        print(f"Epoch {epoch+1}/{config['training']['epochs']}  Loss: {loss:.4f}")
        torch.save(model.state_dict(), checkpoint_dir / f"model_epoch_{epoch}.pt")