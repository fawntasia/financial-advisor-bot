
try:
    from src.data.yfinance_client import YFinanceClient
    print("Successfully imported YFinanceClient")
except ImportError as e:
    print(f"Failed to import YFinanceClient: {e}")
    exit(1)
except Exception as e:
    print(f"Error importing YFinanceClient: {e}")
    exit(1)
