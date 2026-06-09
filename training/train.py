import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from torchvision import transforms

from datasets.multimodal_dataset import MultiModalDataset
from models.model import CLIPCACG
from training.engine import Trainer

def main(df):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = MultiModalDataset(df, tokenizer, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = CLIPCACG(num_classes=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trainer = Trainer(model, optimizer, device)

    for epoch in range(10):
        loss = trainer.train_one_epoch(dataloader)
        print(f"Epoch {epoch} Loss: {loss}")

        torch.save(model.state_dict(), f"outputs/checkpoints/model_epoch_{epoch}.pt")