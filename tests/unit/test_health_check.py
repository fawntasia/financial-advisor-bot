"""
Unit tests for system health checks.
"""
import sys
from contextlib import nullcontext
from unittest.mock import Mock

sys.path.insert(0, ".")

import pytest

from src.utils.health_check import SystemHealth


@pytest.mark.unit
def test_check_database_success():
    cursor = Mock()
    conn = Mock()
    conn.cursor.return_value = cursor
    dal = Mock()
    dal.db_path = Mock()
    dal.db_path.exists.return_value = True
    dal.get_connection.return_value = nullcontext(conn)

    health = SystemHealth(dal=dal)
    result = health.check_database()

    assert result["status"] == "ok"
    assert result["exists"] is True
    assert result["connected"] is True
    cursor.execute.assert_called_once_with("SELECT 1")


@pytest.mark.unit
def test_check_database_error_on_connection():
    dal = Mock()
    dal.db_path = Mock()
    dal.db_path.exists.return_value = False
    dal.get_connection.side_effect = RuntimeError("db down")

    health = SystemHealth(dal=dal)
    result = health.check_database()

    assert result["status"] == "error"
    assert result["connected"] is False
    assert result["error"] == "db down"


@pytest.mark.unit
def test_check_disk_space_ok(monkeypatch):
    gb = 1024 ** 3
    total = 100 * gb
    used = 20 * gb
    free = 80 * gb

    monkeypatch.setattr("src.utils.health_check.shutil.disk_usage", lambda _: (total, used, free))

    health = SystemHealth(dal=Mock())
    result = health.check_disk_space("/")

    assert result["status"] == "ok"
    assert result["total_gb"] == 100.0
    assert result["used_gb"] == 20.0
    assert result["free_gb"] == 80.0
    assert result["percent_free"] == 80.0


@pytest.mark.unit
def test_check_disk_space_warning(monkeypatch):
    gb = 1024 ** 3
    total = 100 * gb
    used = 95 * gb
    free = 5 * gb

    monkeypatch.setattr("src.utils.health_check.shutil.disk_usage", lambda _: (total, used, free))

    health = SystemHealth(dal=Mock())
    result = health.check_disk_space("/")

    assert result["status"] == "warning"
    assert result["percent_free"] == 5.0


@pytest.mark.unit
def test_perform_health_check_summary(monkeypatch):
    health = SystemHealth(dal=Mock())
    monkeypatch.setattr(health, "check_database", lambda: {"status": "ok"})
    monkeypatch.setattr(health, "check_disk_space", lambda: {"status": "ok"})

    result = health.perform_health_check()

    assert result["database"] == {"status": "ok"}
    assert result["disk_space"] == {"status": "ok"}
    assert result["status"] == "ok"
