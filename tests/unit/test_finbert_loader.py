import sys
import types
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")

mock_transformers = types.ModuleType("transformers")
mock_transformers.__dict__["AutoTokenizer"] = types.SimpleNamespace(from_pretrained=MagicMock())
mock_transformers.__dict__["AutoModelForSequenceClassification"] = types.SimpleNamespace(from_pretrained=MagicMock())
sys.modules.setdefault("transformers", mock_transformers)

import pytest
import torch
from types import SimpleNamespace

from src.nlp.finbert_loader import FinBERTLoader


class DummyInputs(dict):
    def to(self, device):
        self["device"] = device
        return self


@pytest.mark.unit
def test_init_loads_model_and_labels():
    tokenizer = MagicMock()
    model = MagicMock()
    model.config = SimpleNamespace(id2label={0: "Neutral", 1: "Positive", 2: "Negative"})

    with patch("src.nlp.finbert_loader.os.path.exists", return_value=True), \
        patch("src.nlp.finbert_loader.torch.cuda.is_available", return_value=False), \
        patch("src.nlp.finbert_loader.AutoTokenizer.from_pretrained", return_value=tokenizer) as mock_tokenizer, \
        patch("src.nlp.finbert_loader.AutoModelForSequenceClassification.from_pretrained", return_value=model) as mock_model:
        loader = FinBERTLoader(model_path="models/finbert")

    assert loader.device == "cpu"
    assert loader.tokenizer is tokenizer
    assert loader.model is model
    assert loader.labels == ["Neutral", "Positive", "Negative"]
    mock_tokenizer.assert_called_once_with("models/finbert")
    mock_model.assert_called_once_with("models/finbert")
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once()


@pytest.mark.unit
def test_predict_returns_labels_and_scores():
    with patch.object(FinBERTLoader, "_load_model", return_value=None):
        loader = FinBERTLoader(model_path="models/finbert")

    loader.device = "cpu"
    loader.labels = ["Neutral", "Positive", "Negative"]

    tokenizer = MagicMock()
    tokenizer.return_value = DummyInputs({"input_ids": [1, 2], "attention_mask": [1, 1]})
    loader.tokenizer = tokenizer

    model = MagicMock()
    model.return_value = SimpleNamespace(logits=torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]]))
    loader.model = model

    probs = torch.tensor([[0.1, 0.7, 0.2], [0.6, 0.2, 0.2]])
    with patch("src.nlp.finbert_loader.torch.nn.functional.softmax", return_value=probs):
        results = loader.predict(["good news", "ok news"], batch_size=2)

    assert results[0]["label"] == "Positive"
    assert results[0]["score"] == pytest.approx(0.7)
    assert results[0]["probs"]["Neutral"] == pytest.approx(0.1)
    assert results[0]["probs"]["Positive"] == pytest.approx(0.7)
    assert results[0]["probs"]["Negative"] == pytest.approx(0.2)

    assert results[1]["label"] == "Neutral"
    assert results[1]["score"] == pytest.approx(0.6)
    assert results[1]["probs"]["Neutral"] == pytest.approx(0.6)
    assert results[1]["probs"]["Positive"] == pytest.approx(0.2)
    assert results[1]["probs"]["Negative"] == pytest.approx(0.2)

    tokenizer.assert_called_once_with(
        ["good news", "ok news"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    model.assert_called_once()


@pytest.mark.unit
def test_get_sentiment_score_maps_labels():
    with patch.object(FinBERTLoader, "_load_model", return_value=None):
        loader = FinBERTLoader(model_path="models/finbert")

    loader.labels = ["Neutral", "Positive", "Negative"]

    with patch.object(
        loader,
        "predict",
        return_value=[
            {"label": "Positive", "score": 0.8, "probs": {}},
            {"label": "Negative", "score": 0.6, "probs": {}},
            {"label": "Neutral", "score": 0.5, "probs": {}},
            {"label": "Error", "score": 0.0, "probs": {}},
        ],
    ):
        scores = loader.get_sentiment_score(["a", "b", "c", "d"])

    assert scores == [0.8, -0.6, 0.0, 0.0]
