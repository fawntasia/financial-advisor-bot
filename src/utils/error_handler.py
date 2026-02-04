"""
Centralized error handling and monitoring.
"""

import sys
import traceback
from typing import Optional, Dict, Any
from datetime import datetime

from src.database.dal import DataAccessLayer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ErrorHandler:
    """
    Centralized error handler for the application.
    Logs errors to both file system (via logger) and database.
    """
    
    def __init__(self, dal: Optional[DataAccessLayer] = None):
        self.dal = dal or DataAccessLayer()
    
    def handle_error(self, error: Exception, component: str, context: Optional[str] = None) -> None:
        """
        Handle an exception by logging it to DB and file.
        
        Args:
            error: The exception object.
            component: Name of the component where error occurred.
            context: Additional context or description.
        """
        # Format the error message and traceback
        error_msg = str(error)
        tb = traceback.format_exc()
        
        full_message = f"{error_msg}"
        if context:
            full_message = f"{context}: {full_message}"
            
        # Log to file
        logger.error(f"Error in {component}: {full_message}\n{tb}")
        
        # Log to database
        try:
            details = f"Traceback:\n{tb}"
            if context:
                details = f"Context: {context}\n{details}"
                
            self.dal.log_system_event(
                level="ERROR",
                component=component,
                message=full_message[:500],  # Truncate if too long
                details=details
            )
        except Exception as db_error:
            # Fallback if DB logging fails
            logger.error(f"Failed to log error to database: {db_error}")

    def log_event(self, level: str, component: str, message: str, details: Optional[str] = None):
        """
        Log a system event to DB and file.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, CRITICAL).
            component: Component name.
            message: Brief message.
            details: Detailed information.
        """
        # Log to file based on level
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(f"[{component}] {message}")
        
        # Log to database
        try:
            self.dal.log_system_event(
                level=level,
                component=component,
                message=message,
                details=details
            )
        except Exception as e:
            logger.error(f"Failed to log event to database: {e}")
