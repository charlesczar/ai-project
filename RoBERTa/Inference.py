import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def probability_inference(model, tokenizer, text):
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1) # Apply softmax to get probabilities from text reviews
    return probabilities #will output example: tensor([[0.1, 0.2, 0.3, 0.4]]) for a each review, where each value corresponds to the probability of each class (deceptive, authentic, irrelevant, vague)
