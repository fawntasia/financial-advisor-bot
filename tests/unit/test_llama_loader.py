import sys
import types
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")

import pytest

from src.llm.llama_loader import LlamaLoader


@pytest.mark.unit
def test_init_missing_model_path_enables_mock_mode():
    with patch("src.llm.llama_loader.os.path.exists", return_value=False), \
        patch.object(LlamaLoader, "_load_model") as mock_load:
        loader = LlamaLoader(model_path="missing/model.gguf")

    assert loader.mock_mode is True
    assert loader.model is None
    mock_load.assert_not_called()


@pytest.mark.unit
def test_detect_gpu_returns_all_layers_when_cuda_available(monkeypatch):
    torch_module = types.ModuleType("torch")
    torch_module.cuda = types.SimpleNamespace(is_available=MagicMock(return_value=True))
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    with patch("src.llm.llama_loader.os.path.exists", return_value=False):
        loader = LlamaLoader(model_path="missing/model.gguf")

    assert loader._detect_gpu() == -1


@pytest.mark.unit
def test_detect_gpu_returns_zero_when_torch_missing():
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with patch("src.llm.llama_loader.os.path.exists", return_value=False):
            loader = LlamaLoader(model_path="missing/model.gguf")

        assert loader._detect_gpu() == 0


@pytest.mark.unit
def test_load_model_uses_cpu_layers_and_sets_model(monkeypatch):
    torch_module = types.ModuleType("torch")
    torch_module.cuda = types.SimpleNamespace(is_available=MagicMock(return_value=False))
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    llama_module = types.ModuleType("llama_cpp")
    llama_instance = MagicMock()
    llama_module.Llama = MagicMock(return_value=llama_instance)
    monkeypatch.setitem(sys.modules, "llama_cpp", llama_module)

    with patch("src.llm.llama_loader.os.path.exists", return_value=True):
        loader = LlamaLoader(model_path="models/llama3/model.gguf", n_ctx=1024)

    assert loader.model is llama_instance
    assert loader.mock_mode is False
    llama_module.Llama.assert_called_once_with(
        model_path="models/llama3/model.gguf",
        n_ctx=1024,
        n_gpu_layers=0,
        verbose=False,
    )


@pytest.mark.unit
def test_load_model_import_error_sets_mock_mode():
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("llama-cpp-python not installed")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with patch("src.llm.llama_loader.os.path.exists", return_value=True):
            loader = LlamaLoader(model_path="models/llama3/model.gguf")

    assert loader.mock_mode is True
    assert loader.model is None
