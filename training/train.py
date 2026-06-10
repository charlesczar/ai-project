import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from torchvision import transforms

from datasets.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
from models.model import CLIPCACG
from training.engine import Trainer

def main(df, config, base_dir):
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Text preprocessing setup
    tokenizer = RobertaTokenizer.from_pretrained(config["model"]["text_model"])

    # 3. Vision preprocessing setup (Added standard normalization)
    image_size = config["data"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # Standard normalization for pretrained vision backbones (CLIP/ImageNet)
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP means
            std=[0.26862954, 0.26130258, 0.27577711]   # CLIP stds
        )
    ])

    # 4. Data Pipeline
    img_dir = str(base_dir / "data" / "images")
    dataset = MultiModalDataset(df, tokenizer, transform, img_dir)
    
    # Check for pin_memory optimization if using GPU
    use_cuda = device.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=multimodal_collate_fn,
        pin_memory=use_cuda,  # Speeds up host-to-device tensor transfers
        drop_last=True        # Prevents batch-size-of-1 errors during evaluation transitions
    )

    print("Train dataset rows (dataset.len):", len(dataset))
    print("Train batches:", len(dataloader), " batch_size:", config["training"]["batch_size"])

    # 5. Model initialization (Make classes dynamic!)
    num_classes = config["model"].get("num_classes", 3) 
    model = CLIPCACG(num_classes=num_classes).to(device)
    
    # 6. Optimizer setup
    # Optimization: Filter out frozen parameters if you freeze backbones later
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=float(config["training"]["lr"]),
        weight_decay=0.01  # AdamW is generally safer for multimodal transformer setups
    )
    
    trainer = Trainer(model, optimizer, device)

    # 7. Checkpoint setup
    checkpoint_dir = base_dir / config["output"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 8. Training loop
    for epoch in range(config["training"]["epochs"]):
        loss = trainer.train_one_epoch(dataloader)
        print(f"Epoch {epoch+1}/{config['training']['epochs']} | Avg Loss: {loss:.4f}")
        
        # Save model weight state
        checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")