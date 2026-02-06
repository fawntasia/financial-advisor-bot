# pyright: reportMissingImports=false
import contextlib
import sys
import types

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")

torch_stub = types.ModuleType("torch")

class DummyCuda:
    @staticmethod
    def is_available():
        return False

@contextlib.contextmanager
def no_grad():
    yield

torch_stub.cuda = DummyCuda()
torch_stub.no_grad = no_grad
torch_stub.nn = types.SimpleNamespace(functional=types.SimpleNamespace(softmax=lambda *args, **kwargs: None))
sys.modules.setdefault("torch", torch_stub)

transformers_stub = types.ModuleType("transformers")

class DummyAuto:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return MagicMock()

transformers_stub.AutoTokenizer = DummyAuto
transformers_stub.AutoModelForSequenceClassification = DummyAuto
sys.modules.setdefault("transformers", transformers_stub)

from src.nlp.sentiment_pipeline import SentimentPipeline

@pytest.fixture
def mock_dal():
    return MagicMock()

@pytest.fixture
def mock_loader():
    return MagicMock()

@pytest.fixture
def pipeline(mock_dal, mock_loader):
    with patch('src.nlp.sentiment_pipeline.FinBERTLoader', return_value=mock_loader):
        return SentimentPipeline(dal=mock_dal)

class TestSentimentPipeline:
    
    def test_process_unprocessed(self, pipeline, mock_dal, mock_loader):
        # Setup mock data
        mock_dal.get_unprocessed_news.return_value = [
            {'id': 1, 'headline': 'Good news', 'ticker': 'AAPL'},
            {'id': 2, 'headline': 'Bad news', 'ticker': 'MSFT'}
        ]
        
        mock_loader.predict.return_value = [
            {'label': 'Positive', 'score': 0.9, 'probs': {'Positive': 0.9, 'Negative': 0.05, 'Neutral': 0.05}},
            {'label': 'Negative', 'score': 0.8, 'probs': {'Positive': 0.1, 'Negative': 0.8, 'Neutral': 0.1}}
        ]
        
        pipeline.process_unprocessed(batch_size=2)
        
        mock_dal.get_unprocessed_news.assert_called_once()
        mock_loader.predict.assert_called_once_with(['Good news', 'Bad news'], batch_size=2)
        mock_dal.bulk_insert_sentiment_scores.assert_called_once()
        
        # Verify stored records
        records = mock_dal.bulk_insert_sentiment_scores.call_args[0][0]
        assert len(records) == 2
        assert records[0]['news_id'] == 1
        assert records[0]['sentiment_label'] == 'Positive'
        assert records[1]['news_id'] == 2
        assert records[1]['sentiment_label'] == 'Negative'

    def test_process_unprocessed_empty(self, pipeline, mock_dal):
        mock_dal.get_unprocessed_news.return_value = []
        pipeline.process_unprocessed()
        mock_dal.bulk_insert_sentiment_scores.assert_not_called()

    def test_update_daily_aggregates(self, pipeline, mock_dal):
        # Setup mock news data for a date
        mock_dal.get_news_for_date.return_value = [
            {'ticker': 'AAPL', 'positive_score': 0.8, 'negative_score': 0.1, 'neutral_score': 0.1, 'confidence': 0.9, 'id': 1},
            {'ticker': 'AAPL', 'positive_score': 0.6, 'negative_score': 0.2, 'neutral_score': 0.2, 'confidence': 0.8, 'id': 2}
        ]
        
        pipeline.update_daily_aggregates("2023-01-01")
        
        mock_dal.get_news_for_date.assert_called_with("2023-01-01")
        mock_dal.insert_daily_sentiment.assert_called_once()
        
        # Verify call arguments
        args = mock_dal.insert_daily_sentiment.call_args[1]
        assert args['ticker'] == 'AAPL'
        assert args['date'] == '2023-01-01'
        assert args['avg_positive'] == 0.7 # (0.8 + 0.6) / 2
        assert args['news_count'] == 2
        assert args['overall_sentiment'] == 'Positive'

    def test_process_date(self, pipeline, mock_dal, mock_loader):
        # Mock connection and cursor for process_date SQL query
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_dal.get_connection.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        mock_cursor.fetchall.return_value = [
            {'id': 1, 'headline': 'Date news', 'ticker': 'AAPL'}
        ]
        
        mock_loader.predict.return_value = [
            {'label': 'Neutral', 'score': 0.7, 'probs': {'Positive': 0.1, 'Negative': 0.2, 'Neutral': 0.7}}
        ]
        
        # We also need to mock update_daily_aggregates dependencies if we don't mock the method itself
        mock_dal.get_news_for_date.return_value = [] # Just to avoid errors in update_daily_aggregates
        
        with patch.object(pipeline, 'update_daily_aggregates') as mock_update:
            pipeline.process_date("2023-01-01")
            mock_update.assert_called_with("2023-01-01")
            
        mock_dal.bulk_insert_sentiment_scores.assert_called_once()

    def test_process_headlines_filters_error_and_defaults(self, pipeline, mock_dal, mock_loader):
        headlines = [
            {'id': 10, 'headline': 'Ok news', 'ticker': 'AAPL'},
            {'id': 11, 'headline': 'Bad parse', 'ticker': 'MSFT'}
        ]

        mock_loader.predict.return_value = [
            {'label': 'Neutral', 'score': 0.55, 'probs': {'Neutral': 0.55}},
            {'label': 'Error', 'score': 0.0, 'probs': {}}
        ]

        pipeline._process_headlines(headlines, batch_size=4)

        mock_loader.predict.assert_called_once_with(['Ok news', 'Bad parse'], batch_size=4)
        mock_dal.bulk_insert_sentiment_scores.assert_called_once()
        records = mock_dal.bulk_insert_sentiment_scores.call_args[0][0]
        assert len(records) == 1
        assert records[0]['news_id'] == 10
        assert records[0]['positive_score'] == 0.0
        assert records[0]['negative_score'] == 0.0
        assert records[0]['neutral_score'] == 0.55
        assert records[0]['confidence'] == 0.55

    def test_process_headlines_all_error_no_insert(self, pipeline, mock_dal, mock_loader):
        headlines = [
            {'id': 20, 'headline': 'Fail 1', 'ticker': 'AAPL'},
            {'id': 21, 'headline': 'Fail 2', 'ticker': 'MSFT'}
        ]

        mock_loader.predict.return_value = [
            {'label': 'Error', 'score': 0.0, 'probs': {}},
            {'label': 'Error', 'score': 0.0, 'probs': {}}
        ]

        pipeline._process_headlines(headlines, batch_size=1)

        mock_loader.predict.assert_called_once_with(['Fail 1', 'Fail 2'], batch_size=1)
        mock_dal.bulk_insert_sentiment_scores.assert_not_called()

    def test_update_daily_aggregates_multiple_tickers(self, pipeline, mock_dal):
        mock_dal.get_news_for_date.return_value = [
            {'ticker': 'AAPL', 'positive_score': 0.9, 'negative_score': 0.05, 'neutral_score': 0.05, 'confidence': 0.8, 'id': 1},
            {'ticker': 'AAPL', 'positive_score': 0.7, 'negative_score': 0.2, 'neutral_score': 0.1, 'confidence': 0.6, 'id': 2},
            {'ticker': 'MSFT', 'positive_score': 0.2, 'negative_score': 0.6, 'neutral_score': 0.2, 'confidence': 0.4, 'id': 3},
            {'ticker': '', 'positive_score': 0.4, 'negative_score': 0.3, 'neutral_score': 0.3, 'confidence': 0.5, 'id': 4}
        ]

        pipeline.update_daily_aggregates("2023-02-02")

        assert mock_dal.insert_daily_sentiment.call_count == 2
        calls = {call.kwargs['ticker']: call.kwargs for call in mock_dal.insert_daily_sentiment.call_args_list}
        assert calls['AAPL']['overall_sentiment'] == 'Positive'
        assert calls['AAPL']['news_count'] == 2
        assert calls['AAPL']['confidence'] == 0.7
        assert calls['MSFT']['overall_sentiment'] == 'Negative'
        assert calls['MSFT']['news_count'] == 1
