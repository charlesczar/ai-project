import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

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
        base = Path(self.img_dir)

        for img_name in image_list:
            # sanitize filename from CSV
            img_name = str(img_name).strip()            # remove whitespace
            img_name = img_name.strip('"').strip("'")   # remove quotes
            img_name = Path(img_name).name              # drop any stray dirs
            path = base / img_name
            if img_name == "" or img_name.lower() in {"nan", "none"}:
                continue

            path = base / img_name

            print("Trying image path:", repr(path))
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            img = Image.open(path).convert("RGB")
            img = self.image_transform(img)
            images.append(img)

        if len(images) == 0:
            raise RuntimeError("No valid images found for this sample.")
        return torch.stack(images)

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


def multimodal_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    # Pad images to (B, N_max, C, H, W) and build a boolean mask
    image_list = [item["images"] for item in batch]
    max_n = max(imgs.shape[0] for imgs in image_list)
    C, H, W = image_list[0].shape[1:]
    B = len(image_list)

    padded = torch.zeros(B, max_n, C, H, W)
    mask = torch.zeros(B, max_n)
    for i, imgs in enumerate(image_list):
        n = imgs.shape[0]
        padded[i, :n] = imgs
        mask[i, :n] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": padded,
        "image_mask": mask,
        "label": labels,
    }