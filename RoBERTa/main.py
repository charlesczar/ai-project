from config import Config
from DataLoad import load_and_tokenize_data
from model import create_model
from train import train_model, evaluate_model


def main():
    config = Config()

    # Load and preprocess data
    tokenized_datasets, tokenizer = load_and_tokenize_data(config)

    # Create model
    model = create_model(config)

    # Train
    trainer = train_model(model, tokenized_datasets, tokenizer, config)

    # FIX: original checked for 'test' split, but load_and_tokenize_data never
    # produces one — it produces 'validation'. Evaluate against validation instead,
    # or against a dedicated test split if you later add one to config.
    if "test" in tokenized_datasets:
        evaluate_model(trainer, tokenized_datasets["test"])
    elif "validation" in tokenized_datasets:
        evaluate_model(trainer, tokenized_datasets["validation"])

    # Save best model and tokenizer
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()