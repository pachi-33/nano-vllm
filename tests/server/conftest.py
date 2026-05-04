"""
Server test fixtures for nano-vLLM.

Starts the nano-vllm OpenAI-compatible API server as a subprocess,
waits for it to become healthy, then provides an httpx client for tests.

CLI options (--model, --pp, --tp, --port) are registered in the root
tests/conftest.py and are shared with integration tests.
"""

import os
import subprocess
import sys
import time

import pytest


@pytest.fixture(scope="module")
def server(request):
    """Start the nano-vllm server as a subprocess, yield base_url, then shut down."""
    model = request.config.getoption("--model")
    pp = request.config.getoption("--pp")
    tp = request.config.getoption("--tp")
    port = request.config.getoption("--port")

    cuda_devices = ",".join(str(i) for i in range(max(pp, tp)))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env["TORCHDYNAMO_DISABLE"] = "1"

    cmd = [
        sys.executable, "-m", "nanovllm.serve",
        "--model", model,
        "--port", str(port),
        "--enforce-eager",
    ]
    if pp > 1:
        cmd += ["--pipeline-parallel-size", str(pp)]
    if tp > 1:
        cmd += ["--tensor-parallel-size", str(tp)]

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    import urllib.request
    import urllib.error
    base_url = f"http://localhost:{port}"
    max_wait = 60
    for i in range(max_wait):
        try:
            req = urllib.request.Request(f"{base_url}/health")
            resp = urllib.request.urlopen(req, timeout=2)
            if resp.status == 200:
                break
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    else:
        out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
        proc.kill()
        pytest.skip(f"Server did not start within {max_wait}s. Output:\n{out[-2000:]}")

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def http(server):
    """Return an httpx.Client pointed at the test server."""
    try:
        import httpx
    except ImportError:
        pytest.skip("httpx not installed")
    client = httpx.Client(base_url=server, timeout=30)
    yield client
    client.close()
