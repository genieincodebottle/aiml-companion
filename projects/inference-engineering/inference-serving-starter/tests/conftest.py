"""Shared fixtures. GIVEN."""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def start_engine():
    """start_engine("--gpu", "l4", ...) -> base_url. Servers are stopped at teardown."""
    procs: list = []

    def _start(*args: str, time_scale: float = 50.0) -> str:
        port = _free_port()
        p = subprocess.Popen([sys.executable, "-m", "engine_sim", "--port", str(port), "--time-scale",
                              str(time_scale), *args], cwd=ROOT, stdout=subprocess.DEVNULL)
        procs.append(p)
        base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200:
                    return base
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("engine_sim did not start")

    yield _start
    for p in procs:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
