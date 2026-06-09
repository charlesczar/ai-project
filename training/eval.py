import torch
from sklearn.metrics import classification_report

def evaluate(model, dataloader, device):

    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)

            outputs = model(input_ids, attention_mask, image)

            pred = outputs.argmax(dim=1).cpu().numpy()

            preds.extend(pred)
            labels.extend(batch["label"].numpy())

    print(classification_report(labels, preds))