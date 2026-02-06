import importlib
import sys
import types

sys.path.insert(0, ".")


class DummySpinner:
    def __init__(self, calls, label):
        self.calls = calls
        self.label = label

    def __enter__(self):
        self.calls.append(("spinner.enter", self.label))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.calls.append(("spinner.exit", self.label))
        return False


class DummyStatus:
    def __init__(self, calls, label, expanded):
        self.calls = calls
        self.label = label
        self.expanded = expanded

    def __enter__(self):
        self.calls.append(("status.enter", self.label, self.expanded))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.calls.append(("status.exit", self.label, self.expanded))
        return False

    def update(self, **kwargs):
        self.calls.append(("status.update", kwargs))


class DummyColumn:
    def __init__(self, calls, index):
        self.calls = calls
        self.index = index

    def metric(self, label, value, delta=None, help=None):
        self.calls.append(("column.metric", self.index, label, value, delta, help))


class SessionState:
    def __init__(self):
        self._data = {}

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]


class FakeStreamlit(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.session_state = SessionState()

    def header(self, text):
        self.calls.append(("header", text))

    def write(self, text):
        self.calls.append(("write", text))

    def info(self, text):
        self.calls.append(("info", text))

    def subheader(self, text, help=None):
        self.calls.append(("subheader", text, help))

    def image(self, url, use_container_width=False):
        self.calls.append(("image", url, use_container_width))

    def spinner(self, label):
        self.calls.append(("spinner", label))
        return DummySpinner(self.calls, label)

    def status(self, label, expanded=False):
        self.calls.append(("status", label, expanded))
        return DummyStatus(self.calls, label, expanded)

    def columns(self, count):
        self.calls.append(("columns", count))
        return [DummyColumn(self.calls, index) for index in range(count)]

    def markdown(self, text):
        self.calls.append(("markdown", text))

    def caption(self, text):
        self.calls.append(("caption", text))

    def error(self, text):
        self.calls.append(("error", text))


def _load_views(fake_streamlit):
    sys.modules["streamlit"] = fake_streamlit
    if "src.ui.views" in sys.modules:
        return importlib.reload(sys.modules["src.ui.views"])
    return importlib.import_module("src.ui.views")


def test_show_stock_analysis_uses_default_ticker():
    fake_st = FakeStreamlit()
    views = _load_views(fake_st)

    views.show_stock_analysis()

    assert any(
        call[0] == "header" and "Stock Analysis" in call[1]
        for call in fake_st.calls
    )
    assert ("spinner", "Fetching data for AAPL...") in fake_st.calls
    assert ("subheader", "Technical Overview", "A visual representation of stock price movements and technical indicators.") in fake_st.calls
    assert ("image", "https://via.placeholder.com/800x400.png?text=Stock+Chart+Placeholder", True) in fake_st.calls


def test_show_stock_analysis_uses_session_ticker():
    fake_st = FakeStreamlit()
    fake_st.session_state["ticker"] = "MSFT"
    views = _load_views(fake_st)

    views.show_stock_analysis()

    assert ("spinner", "Fetching data for MSFT...") in fake_st.calls
    assert any(
        call[0] == "write" and "MSFT" in call[1]
        for call in fake_st.calls
    )


def test_show_chat_interface_with_manager():
    fake_st = FakeStreamlit()
    chat_calls = []

    class DummyChatManager:
        def display_chat(self):
            chat_calls.append("displayed")

    fake_st.session_state["chat_manager"] = DummyChatManager()
    views = _load_views(fake_st)

    views.show_chat_interface()

    assert chat_calls == ["displayed"]
    assert not any(call[0] == "error" for call in fake_st.calls)


def test_show_chat_interface_without_manager():
    fake_st = FakeStreamlit()
    views = _load_views(fake_st)

    views.show_chat_interface()

    assert ("error", "Chat Manager not initialized. Please check the application setup.") in fake_st.calls


def test_show_dashboard_updates_status_and_metrics():
    fake_st = FakeStreamlit()
    views = _load_views(fake_st)

    views.show_dashboard()

    assert any(
        call[0] == "header" and "Market Dashboard" in call[1]
        for call in fake_st.calls
    )
    assert ("status", "Loading market data...", True) in fake_st.calls
    assert ("status.update", {"label": "Market Data Loaded", "state": "complete", "expanded": False}) in fake_st.calls
    assert ("columns", 3) in fake_st.calls
    metric_calls = [call for call in fake_st.calls if call[0] == "column.metric"]
    assert [call[2] for call in metric_calls] == ["S&P 500", "Market Volatility", "Sentiment Score"]


def test_show_disclaimer_renders_text():
    fake_st = FakeStreamlit()
    views = _load_views(fake_st)

    views.show_disclaimer()

    assert ("markdown", "---") in fake_st.calls
    assert any(
        call[0] == "caption" and "Mandatory Financial Disclaimer" in call[1]
        for call in fake_st.calls
    )
