from transformers import AutoModelForSequenceClassification

def create_model(config):    
    model = AutoModelForSequenceClassification.from_pretrained(        
        config.model_name,        
        num_labels=config.num_labels    
    )    
    return model
