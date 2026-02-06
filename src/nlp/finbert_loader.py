import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union
import logging

logger = logging.getLogger(__name__)

class FinBERTLoader:
    """
    A class to load and use the FinBERT model for sentiment analysis locally.
    """
    def __init__(self, model_path: str = "models/finbert"):
        """
        Initializes the FinBERT loader.
        
        Args:
            model_path: Path to the local directory containing the model and tokenizer.
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.labels = ["Neutral", "Positive", "Negative"] # Default for yiyanghkust/finbert-tone
        
        self._load_model()

    def _load_model(self):
        """
        Loads the model and tokenizer from the specified path.
        """
        if not os.path.exists(self.model_path):
            logger.error(f"Model path '{self.model_path}' does not exist. Please run 'scripts/download_finbert.py' first.")
            return

        try:
            logger.info(f"Loading FinBERT model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Update labels from config if available
            if hasattr(self.model.config, "id2label"):
                id2label = self.model.config.id2label
                self.labels = [id2label[i] for i in range(len(id2label))]
                
            logger.info("FinBERT model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            self.tokenizer = None
            self.model = None

    def predict(self, texts: Union[str, List[str]], batch_size: int = 16) -> List[Dict[str, Union[str, float]]]:
        """
        Predicts sentiment for a given text or list of texts.
        
        Args:
            texts: A single string or a list of strings to analyze.
            batch_size: Size of batches for processing.
            
        Returns:
            A list of dictionaries containing sentiment label and score.
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not loaded.")
            return []

        if isinstance(texts, str):
            texts = [texts]

        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            try:
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                for prob in probs:
                    label_idx = torch.argmax(prob).item()
                    results.append({
                        "label": self.labels[label_idx],
                        "score": prob[label_idx].item(),
                        "probs": {self.labels[j]: prob[j].item() for j in range(len(self.labels))}
                    })
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                for _ in range(len(batch_texts)):
                    results.append({"label": "Error", "score": 0.0, "probs": {}})

        return results

    def get_sentiment_score(self, texts: Union[str, List[str]]) -> List[float]:
        """
        Returns a single numeric sentiment score between -1 and 1.
        (1 for Positive, 0 for Neutral, -1 for Negative)
        
        Args:
            texts: A single string or a list of strings.
            
        Returns:
            A list of sentiment scores.
        """
        predictions = self.predict(texts)
        scores = []
        
        # Mapping labels to numeric values
        # Assumes labels contain 'Positive', 'Negative', and 'Neutral' (case-insensitive)
        label_map = {}
        for i, label in enumerate(self.labels):
            l_lower = label.lower()
            if "positive" in l_lower:
                label_map[label] = 1.0
            elif "negative" in l_lower:
                label_map[label] = -1.0
            else:
                label_map[label] = 0.0
                
        for pred in predictions:
            if pred["label"] == "Error":
                scores.append(0.0)
            else:
                scores.append(label_map.get(pred["label"], 0.0) * pred["score"])
                
        return scores
