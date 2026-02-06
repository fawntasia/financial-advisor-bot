import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def download_model(model_name="yiyanghkust/finbert-tone", save_path="models/finbert"):
    """
    Downloads the FinBERT model and tokenizer from HuggingFace and saves them locally.
    """
    print(f"Downloading model '{model_name}' to '{save_path}'...")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")

    try:
        # Download and save tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        
        # Download and save model
        print("Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.save_pretrained(save_path)
        
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def verify_model(save_path="models/finbert"):
    """
    Verifies that the saved model can be loaded and performs a simple inference.
    """
    print(f"Verifying model at '{save_path}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        model = AutoModelForSequenceClassification.from_pretrained(save_path)
        
        inputs = tokenizer(["Stocks rebounded today", "Earnings were disappointing"], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        print("Inference successful.")
        print(f"Predictions shape: {predictions.shape}")
        
        # Mapping for yiyanghkust/finbert-tone: 0: Neutral, 1: Positive, 2: Negative
        labels = ["Neutral", "Positive", "Negative"]
        for i, text in enumerate(["Stocks rebounded today", "Earnings were disappointing"]):
            label_idx = torch.argmax(predictions[i]).item()
            print(f"Text: '{text}' -> Sentiment: {labels[label_idx]} ({predictions[i][label_idx]:.4f})")
            
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False

if __name__ == "__main__":
    # Ensure we are in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    model_path = os.path.join("models", "finbert")
    
    if download_model(save_path=model_path):
        if verify_model(save_path=model_path):
            print("Model setup successfully!")
        else:
            print("Model verification failed.")
            sys.exit(1)
    else:
        print("Model download failed.")
        sys.exit(1)
