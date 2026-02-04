"""
Example tests to verify pytest setup is working.
"""
import sys
sys.path.insert(0, '.')

def test_example():
    """Simple test to verify pytest is working."""
    assert True

def test_imports():
    """Test that key imports work."""
    from src.data.stock_data import StockDataProcessor
    assert StockDataProcessor is not None
