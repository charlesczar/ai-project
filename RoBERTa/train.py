import numpy as np
from transformers import Trainer, TrainingArguments


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)

    # Per-class precision, recall, F1 (binary classification)
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train_model(model, datasets, tokenizer, config):
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        eval_strategy="epoch",        # FIX: was evaluation_strategy (removed in transformers 5.x)
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=1,              # FIX: was 50 — too large for small datasets (only ~2 steps/epoch)
        save_total_limit=1,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        processing_class=tokenizer,   # FIX: was tokenizer= (removed in transformers 5.x)
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer


def evaluate_model(trainer, dataset):
    results = trainer.evaluate(eval_dataset=dataset)
    print("Evaluation results:", results)
    return results