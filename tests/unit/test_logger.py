import sys
import importlib
import logging

import pytest

sys.path.insert(0, ".")

EXPECTED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class DummyFileHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs


def _reload_logger_module(monkeypatch):
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    root_logger.handlers[:] = []
    monkeypatch.setattr(logging, "FileHandler", DummyFileHandler)

    if "src.utils.logger" in sys.modules:
        del sys.modules["src.utils.logger"]

    import src.utils.logger as logger_module
    importlib.reload(logger_module)

    return logger_module, original_handlers, original_level


@pytest.fixture
def logger_module(monkeypatch):
    logger_module, original_handlers, original_level = _reload_logger_module(
        monkeypatch
    )
    yield logger_module

    root_logger = logging.getLogger()
    root_logger.handlers[:] = original_handlers
    root_logger.setLevel(original_level)


def test_logging_configuration(logger_module):
    root_logger = logging.getLogger()

    assert root_logger.level == logging.INFO
    configured_handlers = [
        handler
        for handler in root_logger.handlers
        if isinstance(handler, (DummyFileHandler, logging.StreamHandler))
    ]
    assert len(configured_handlers) >= 2

    dummy_handler = next(
        handler for handler in configured_handlers if isinstance(handler, DummyFileHandler)
    )
    stream_handler = next(
        handler for handler in configured_handlers if isinstance(handler, logging.StreamHandler)
    )
    for handler in (dummy_handler, stream_handler):
        assert handler.formatter is not None
        assert handler.formatter._fmt == EXPECTED_FORMAT
    assert dummy_handler.args
    assert str(dummy_handler.args[0]).endswith(".log")


def test_get_logger_returns_named_logger(logger_module):
    logger = logger_module.get_logger("unit-test")
    assert logger.name == "unit-test"


def test_no_duplicate_handlers_on_reload(monkeypatch):
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    try:
        root_logger.handlers[:] = []
        monkeypatch.setattr(logging, "FileHandler", DummyFileHandler)

        if "src.utils.logger" in sys.modules:
            del sys.modules["src.utils.logger"]

        import src.utils.logger as logger_module
        importlib.reload(logger_module)
        first_count = len(root_logger.handlers)

        importlib.reload(logger_module)
        second_count = len(root_logger.handlers)

        assert second_count == first_count
    finally:
        root_logger.handlers[:] = original_handlers
        root_logger.setLevel(original_level)
