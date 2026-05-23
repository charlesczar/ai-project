from datasets import load_dataset #replace with actual dataset name
from transformers import AutoTokenizer

def load_and_tokenize_data(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset = load_dataset(config.dataset_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=config.max_length,
            return_attention_mask=True
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets, tokenizer