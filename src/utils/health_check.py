"""
System health checks.
"""

import shutil
import logging
from pathlib import Path
from typing import Dict, Any

from src.database.dal import DataAccessLayer

logger = logging.getLogger(__name__)

class SystemHealth:
    """System health check utilities."""
    
    def __init__(self, dal: DataAccessLayer = None):
        self.dal = dal or DataAccessLayer()
        self.db_path = self.dal.db_path
        
    def check_database(self) -> Dict[str, Any]:
        """Check database connection and file existence."""
        result = {
            "status": "unknown",
            "exists": False,
            "connected": False,
            "error": None
        }
        
        try:
            # Check file existence
            if self.db_path.exists():
                result["exists"] = True
                
            # Check connection by running a simple query
            with self.dal.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result["connected"] = True
                result["status"] = "ok"
                
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"Health check failed for database: {e}")
            
        return result
    
    def check_disk_space(self, path: str = ".") -> Dict[str, Any]:
        """Check available disk space."""
        result = {
            "status": "unknown",
            "total_gb": 0,
            "used_gb": 0,
            "free_gb": 0,
            "percent_free": 0
        }
        
        try:
            total, used, free = shutil.disk_usage(path)
            
            # Convert to GB
            gb = 1024 ** 3
            result["total_gb"] = round(total / gb, 2)
            result["used_gb"] = round(used / gb, 2)
            result["free_gb"] = round(free / gb, 2)
            result["percent_free"] = round((free / total) * 100, 1)
            result["status"] = "ok"
            
            if result["percent_free"] < 10:
                result["status"] = "warning"
                logger.warning(f"Low disk space: {result['percent_free']}% free")
                
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"Health check failed for disk space: {e}")
            
        return result
        
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform all health checks."""
        return {
            "database": self.check_database(),
            "disk_space": self.check_disk_space(),
            "status": "ok"  # Aggregate status logic could be added here
        }

if __name__ == "__main__":
    # Run health check when script is executed directly
    logging.basicConfig(level=logging.INFO)
    health = SystemHealth()
    print(health.perform_health_check())
