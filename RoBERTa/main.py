from config import Config
from DataLoad import load_and_tokenize_data
from model import create_model
from train import train_model

def main():
    config = Config()
    
    # Load and preprocess data
    tokenized_datasets, tokenizer = load_and_tokenize_data(config)
    
    # Create model
    model = create_model(config)
    
    # Train
    trainer = train_model(model, tokenized_datasets, config)
    
    # Evaluate
    if 'test' in tokenized_datasets:
        evaluate_model(trainer, tokenized_datasets['test'])
    
    # Save model
    model.save_pretrained('./final_model')
    tokenizer.save_pretrained('./final_model')

if __name__ == '__main__':
    main()