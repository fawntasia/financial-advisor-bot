import sys

import pytest

sys.path.insert(0, ".")

from src.utils.retry import retry_with_backoff


@pytest.mark.unit
def test_retry_success_after_failures(monkeypatch):
    attempts = {"count": 0}
    delays = []

    def fake_sleep(delay):
        delays.append(delay)

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("temporary")
        return "ok"

    monkeypatch.setattr("time.sleep", fake_sleep)
    decorated = retry_with_backoff(
        max_attempts=5,
        base_delay=1.0,
        max_delay=10.0,
        exceptions=ValueError,
    )(flaky)

    assert decorated() == "ok"
    assert attempts["count"] == 3
    assert delays == [1.0, 2.0]


@pytest.mark.unit
def test_retry_backoff_caps_at_max_delay(monkeypatch):
    attempts = {"count": 0}
    delays = []

    def fake_sleep(delay):
        delays.append(delay)

    def always_fail():
        attempts["count"] += 1
        raise RuntimeError("nope")

    monkeypatch.setattr("time.sleep", fake_sleep)
    decorated = retry_with_backoff(
        max_attempts=4,
        base_delay=2.0,
        max_delay=5.0,
        exceptions=RuntimeError,
    )(always_fail)

    with pytest.raises(RuntimeError):
        decorated()

    assert attempts["count"] == 4
    assert delays == [2.0, 4.0, 5.0]


@pytest.mark.unit
def test_retry_raises_after_max_attempts(monkeypatch):
    attempts = {"count": 0}
    delays = []

    def fake_sleep(delay):
        delays.append(delay)

    def always_fail():
        attempts["count"] += 1
        raise ValueError("boom")

    monkeypatch.setattr("time.sleep", fake_sleep)
    decorated = retry_with_backoff(
        max_attempts=3,
        base_delay=0.5,
        max_delay=2.0,
        exceptions=ValueError,
    )(always_fail)

    with pytest.raises(ValueError):
        decorated()

    assert attempts["count"] == 3
    assert delays == [0.5, 1.0]
