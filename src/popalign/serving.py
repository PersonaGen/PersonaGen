"""Local vLLM server lifecycle helpers."""

from __future__ import annotations

import subprocess
import socket
import sys
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import requests


def wait_for_server(port: int, timeout: int = 600, process: Optional[subprocess.Popen] = None) -> None:
    url = f"http://localhost:{port}/v1/models"
    start = time.monotonic()
    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM exited before becoming ready (exit code {process.returncode}).")
        try:
            if requests.get(url, timeout=3).status_code == 200:
                if process is not None and process.poll() is not None:
                    raise RuntimeError(f"vLLM exited during startup (exit code {process.returncode}).")
                return
        except requests.RequestException:
            pass
        if time.monotonic() - start > timeout:
            raise RuntimeError(f"vLLM server did not start within {timeout}s on port {port}.")
        time.sleep(2)


def _ensure_port_available(port: int) -> None:
    # Check before starting a child: a pre-existing server must never satisfy
    # the readiness probe for the model we are trying to launch.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError as exc:
            raise RuntimeError(
                f"Port {port} is already in use. Choose another model.port, or omit "
                "model.path and configure model.api_base to use an existing endpoint."
            ) from exc


def start_vllm_server(
    model_path: str,
    model_name: str,
    port: int,
    gpu: int,
    timeout: int = 3000,
    phi: bool = False,
) -> subprocess.Popen:
    _ensure_port_available(port)
    parallel_arg = "--pipeline-parallel-size" if phi else "--tensor-parallel-size"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        f"--model={model_path}",
        f"--served-model-name={model_name}",
        f"{parallel_arg}={gpu}",
        "--gpu-memory-utilization=0.85",
        f"--port={port}",
    ]
    if not phi:
        command.extend(["--trust-remote-code", "--enforce-eager"])

    print(f"[INFO] Launching vLLM: {' '.join(command)}")
    process = subprocess.Popen(command, shell=False)
    try:
        wait_for_server(port, timeout=timeout, process=process)
    except BaseException:
        stop_process(process)
        raise
    return process


def stop_process(process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


@contextmanager
def maybe_vllm_server(model: Dict, runtime: Dict) -> Iterator[None]:
    """Start vLLM when `model.path` is provided, otherwise use an existing endpoint."""
    model_path = model.get("path")
    process = None
    try:
        if model_path:
            model_name = model["name"]
            port = int(model.get("port", 8000))
            gpu = int(model.get("gpu", runtime.get("gpu", 1)))
            process = start_vllm_server(
                model_path=model_path,
                model_name=model_name,
                port=port,
                gpu=gpu,
                phi="phi" in model_name.lower(),
            )
        yield
    finally:
        stop_process(process)
