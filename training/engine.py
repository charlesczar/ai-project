import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            image = batch["images"].to(self.device)
            image_mask = batch["image_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids, attention_mask, image, image_mask)

            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)