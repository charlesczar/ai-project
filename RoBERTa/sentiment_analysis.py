import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config
import os
import csv


def load_sentiment_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()  # FIX: set model to eval mode immediately after loading
    return tokenizer, model


def predict_sentiment(text, tokenizer, model, config=None):
    # FIX: accept config as a parameter instead of creating a new Config() on
    # every call — avoids re-reading defaults each inference and makes it
    # easy to pass a custom config if needed.
    if config is None:
        config = Config()

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_id = int(outputs.logits.argmax(dim=-1).item())
    if config.num_labels == 2:
        label_map = {0: "negative", 1: "positive"}
    else:
        label_map = {i: str(i) for i in range(config.num_labels)}
    return label_map.get(predicted_id, str(predicted_id))


if __name__ == "__main__":
    cfg = Config()
    tokenizer, model = load_sentiment_model(cfg.output_dir)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "validation.csv")
    try:
        # 3. Open the dynamic path instead of the hardcoded string
        with open(csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row["text"]
                true_label = row["label"]
                
                prediction = predict_sentiment(text, tokenizer, model, config=cfg)
                
                print(f"Review: {text}")
                print(f"Actual Label: {true_label} | Model Predicted: {prediction}")
                print("-" * 50)
                
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_path}'.")