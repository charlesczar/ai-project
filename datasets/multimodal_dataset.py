import torch
from torch.utils.data import Dataset
from PIL import Image

class MultiModalDataset(Dataset):
    def __init__(self, df, tokenizer, image_transform, img_dir):
        self.df = df
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def load_images(self, image_list):
        images = []

        for img_name in image_list:
            path = f"{self.img_dir}/{img_name}"
            img = Image.open(path).convert("RGB")
            img = self.image_transform(img)
            images.append(img)

        images = torch.stack(images)  # (N, C, H, W)
        return images

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- TEXT MERGE ----
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

        # ---- IMAGE PROCESSING ----
        image_files = str(row["image_paths"]).split(";")
        images = self.load_images(image_files)

        return {
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "images": images,  # multiple images tensor
            "label": torch.tensor(row["label"], dtype=torch.long)
        }