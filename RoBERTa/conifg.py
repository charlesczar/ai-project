from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = 'roberta-base'
    dataset_name: str = 'your_dataset_name'  # Replace with actual dataset
    max_length: int = 512 #max length of text in reviews, adjust as needed
    batch_size: int = 32 #batch size for training change as needed
    learning_rate: float = 2e-5 #learning rate for training
    num_epochs: int = 3 
    num_labels: int = 4  # Adjust based on your task
    