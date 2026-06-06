from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize_data(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # FIX: original code always resolved paths relative to DataLoad.py's directory,
    # which silently breaks when you run main.py from a different working directory.
    # Now we check CWD first, then fall back to the script's directory.
    def resolve_path(filename):
        p = Path(filename)
        if p.is_absolute():
            return p
        cwd_path = Path.cwd() / filename
        if cwd_path.exists():
            return cwd_path
        return Path(__file__).resolve().parent / filename

    train_path = resolve_path(config.train_file)
    validation_path = resolve_path(config.validation_file)

    if validation_path.exists():
        dataset = load_dataset(
            "csv",
            data_files={"train": str(train_path), "validation": str(validation_path)},
            delimiter=",",
        )
    else:
        dataset = load_dataset(
            "csv",
            data_files=str(train_path),
            delimiter=",",
            split="train",
        )
        split_data = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
        dataset = {"train": split_data["train"], "validation": split_data["test"]}

    def tokenize_function(examples):
        return tokenizer(
            examples[config.text_column],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_attention_mask=True,
        )

    if isinstance(dataset, dict):
        tokenized_datasets = {
            split: ds.map(tokenize_function, batched=True)
            for split, ds in dataset.items()
        }
    else:
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

    if config.label_column != "labels":
        def rename_label(examples):
            return {"labels": examples[config.label_column]}

        if isinstance(tokenized_datasets, dict):
            tokenized_datasets = {
                split: ds.map(rename_label, batched=True)
                for split, ds in tokenized_datasets.items()
            }
        else:
            tokenized_datasets = tokenized_datasets.map(rename_label, batched=True)

    def clean_columns(ds):
        columns_to_remove = [
            column
            for column in ds.column_names
            if column not in ["input_ids", "attention_mask", "labels"]
        ]
        return ds.remove_columns(columns_to_remove)

    if isinstance(tokenized_datasets, dict):
        tokenized_datasets = {
            split: clean_columns(ds)
            for split, ds in tokenized_datasets.items()
        }
    else:
        tokenized_datasets = clean_columns(tokenized_datasets)

    return tokenized_datasets, tokenizer