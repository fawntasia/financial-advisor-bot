import os
import subprocess
import sys
import time
import urllib.request

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync.sync_playwright


def _wait_for_streamlit(url: str, timeout: int = 30) -> None:
    deadline = time.time() + timeout
    last_exc = None

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5)

    raise RuntimeError(f"Streamlit server not ready at {url}: {last_exc}")


def _start_streamlit(project_root: str, port: int) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--browser.gatherUsageStats",
        "false",
    ]

    return subprocess.Popen(
        command,
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_streamlit(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@pytest.mark.integration
@pytest.mark.slow
def test_full_app_playwright_flow():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    url = "http://127.0.0.1:8501"
    evidence_path = os.path.join(
        project_root,
        ".sisyphus",
        "evidence",
        "task-27-full-app.png",
    )
    os.makedirs(os.path.dirname(evidence_path), exist_ok=True)

    process = _start_streamlit(project_root, 8501)
    try:
        _wait_for_streamlit(url)

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded")
            page.get_by_label("Enter Ticker (e.g., TSLA, MSFT)").fill("MSFT")
            page.keyboard.press("Enter")
            page.wait_for_timeout(1000)
            page.screenshot(path=evidence_path, full_page=True)
            browser.close()
    finally:
        _stop_streamlit(process)
