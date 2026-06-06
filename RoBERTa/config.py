import os
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    model_name: str = 'roberta-base'
    train_file: str = 'train.csv'
    validation_file: str = 'validation.csv'
    text_column: str = 'text'
    label_column: str = 'label'
    max_length: int = 128
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    num_labels: int = 2
    

    output_dir: str = os.path.join(BASE_DIR, 'final_model')