"""
Unit tests for error handling, retry logic, and health checks.
"""

import unittest
from unittest.mock import MagicMock, patch
import time
import logging

from src.utils.error_handler import ErrorHandler
from src.utils.retry import retry_with_backoff
from src.utils.health_check import SystemHealth

class TestErrorHandler(unittest.TestCase):
    def setUp(self):
        self.mock_dal = MagicMock()
        self.error_handler = ErrorHandler(dal=self.mock_dal)
        
    @patch("src.utils.error_handler.logger")
    def test_handle_error(self, mock_logger):
        error = ValueError("Test error")
        self.error_handler.handle_error(error, "TestComponent", "Context info")
        
        # Verify logger was called
        mock_logger.error.assert_called()
        
        # Verify DAL was called
        self.mock_dal.log_system_event.assert_called_with(
            level="ERROR",
            component="TestComponent",
            message="Context info: Test error",
            details=unittest.mock.ANY
        )

    @patch("src.utils.error_handler.logger")
    def test_log_event(self, mock_logger):
        self.error_handler.log_event("INFO", "TestComponent", "Test message")
        
        # Verify logger was called
        mock_logger.info.assert_called_with("[TestComponent] Test message")
        
        # Verify DAL was called
        self.mock_dal.log_system_event.assert_called_with(
            level="INFO",
            component="TestComponent",
            message="Test message",
            details=None
        )

class TestRetry(unittest.TestCase):
    def test_retry_success(self):
        mock_func = MagicMock(return_value="success")
        
        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def decorated_func():
            return mock_func()
            
        result = decorated_func()
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 1)

    def test_retry_fail_then_succeed(self):
        mock_func = MagicMock(side_effect=[ValueError("Fail 1"), "success"])
        
        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def decorated_func():
            return mock_func()
            
        result = decorated_func()
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 2)

    def test_retry_exhausted(self):
        mock_func = MagicMock(side_effect=ValueError("Persistent fail"))
        
        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def decorated_func():
            return mock_func()
            
        with self.assertRaises(ValueError):
            decorated_func()
        self.assertEqual(mock_func.call_count, 3)

class TestHealthCheck(unittest.TestCase):
    def setUp(self):
        self.mock_dal = MagicMock()
        self.health = SystemHealth(dal=self.mock_dal)
        
    def test_check_database_connected(self):
        # Mock successful connection context manager
        mock_conn = MagicMock()
        self.mock_dal.get_connection.return_value.__enter__.return_value = mock_conn
        self.health.db_path = MagicMock()
        self.health.db_path.exists.return_value = True
        
        result = self.health.check_database()
        
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["exists"])
        self.assertTrue(result["connected"])

    @patch("src.utils.health_check.shutil.disk_usage")
    def test_check_disk_space(self, mock_disk_usage):
        # Total, Used, Free (in bytes)
        # 100 GB total, 50 GB used, 50 GB free
        gb = 1024**3
        mock_disk_usage.return_value = (100 * gb, 50 * gb, 50 * gb)
        
        result = self.health.check_disk_space()
        
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total_gb"], 100.0)
        self.assertEqual(result["percent_free"], 50.0)

if __name__ == "__main__":
    unittest.main()
