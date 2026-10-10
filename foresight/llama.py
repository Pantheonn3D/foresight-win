"""llama.cpp runtime: fetches the prebuilt Windows binaries, runs llama-server
processes, and wraps the HTTP calls the app needs (transcribe, chat, embed)."""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Iterator

import httpx

from . import config, models

log = logging.getLogger(__name__)

# Pinned so a llama.cpp release can't change behaviour under the user. Bump
# deliberately after the CI self-test passes on the new tag.
LLAMA_CPP_TAG = os.environ.get("FORESIGHT_LLAMA_TAG", "b11552")
GH = "https://github.com/ggml-org/llama.cpp/releases/download"
CUDA = "12.4"  # 12.x runtime works with any driver from 527.41 up; 13.x needs a much newer driver

EXE = "llama-server.exe" if sys.platform == "win32" else "llama-server"


def detect_backend() -> str:
    if sys.platform != "win32":
        return "cpu"
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        if out.returncode == 0 and "GPU" in out.stdout:
            return "cuda"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "vulkan"


def backend() -> str:
    b = config.settings().backend
    return detect_backend() if b == "auto" else b


def _assets(b: str) -> list[str]:
    if b == "cuda":
        return [f"llama-{LLAMA_CPP_TAG}-bin-win-cuda-{CUDA}-x64.zip", f"cudart-llama-bin-win-cuda-{CUDA}-x64.zip"]
    return [f"llama-{LLAMA_CPP_TAG}-bin-win-{b}-x64.zip"]


def runtime_download_size() -> int | None:
    total = 0
    try:
        with httpx.Client(follow_redirects=True, timeout=20) as c:
            for name in _assets(backend()):
                r = c.head(f"{GH}/{LLAMA_CPP_TAG}/{name}")
                r.raise_for_status()
                total += int(r.headers.get("content-length", 0))
    except httpx.HTTPError as e:
        log.warning("could not size llama.cpp download: %s", e)
        return None
    return total


def runtime_dir(b: str | None = None) -> Path:
    return config.bin_dir() / f"{LLAMA_CPP_TAG}-{b or backend()}"


def find_server(b: str | None = None) -> Path | None:
    override = os.environ.get("FORESIGHT_LLAMA_SERVER")
    if override:
        return Path(override)
    root = runtime_dir(b)
    if not root.exists():
        return None
    hits = list(root.rglob(EXE))
    return hits[0] if hits else None


def runtime_installed() -> bool:
    return find_server() is not None


def install_runtime(progress: models.ProgressFn | None = None, cancel: threading.Event | None = None) -> None:
    b = backend()
    dest = runtime_dir(b)
    staging = dest.with_name(dest.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    names = _assets(b)
    for i, name in enumerate(names):
        z = staging / name
        models.download_file(f"{GH}/{LLAMA_CPP_TAG}/{name}", z, 0, None,
                             (lambda d, t, i=i: progress(i * 1000 + int(1000 * d / max(t, 1)), 1000 * len(names)))
                             if progress else None, cancel)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(staging)
        z.unlink()
    # cudart DLLs must sit beside llama-server.exe
    exe = next(staging.rglob(EXE), None)
    if exe is None:
        raise RuntimeError(f"{EXE} missing from llama.cpp {LLAMA_CPP_TAG} {b} build")
    for dll in staging.glob("*.dll"):
        if dll.parent != exe.parent and not (exe.parent / dll.name).exists():
            shutil.move(str(dll), exe.parent / dll.name)
    shutil.rmtree(dest, ignore_errors=True)
    staging.rename(dest)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """One llama-server child process."""

    def __init__(self, name: str, args: list[str]):
        self.name = name
        self.args = args
        self.port = 0
        self.proc: subprocess.Popen | None = None
        self.log_path = config.logs_dir() / f"llama-{name}.log"
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, timeout: float = 300) -> None:
        with self._lock:
            if self.running():
                return
            exe = find_server()
            if exe is None:
                raise RuntimeError("llama.cpp runtime is not installed")
            self.port = _free_port()
            cmd = [str(exe), *self.args, "--host", "127.0.0.1", "--port", str(self.port), "--no-webui"]
            log.info("starting %s: %s", self.name, " ".join(cmd))
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            logf = open(self.log_path, "w", encoding="utf-8", errors="replace")
            self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=exe.parent,
                                         creationflags=flags)
            logf.close()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} server exited with code {self.proc.returncode}:\n{self.tail()}")
            try:
                if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"{self.name} server did not become ready in {timeout:.0f}s:\n{self.tail()}")

    def stop(self) -> None:
        with self._lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.proc = None

    def tail(self, n: int = 30) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])


ASR_PROMPT = (
    "Transcribe the following speech segment in {lang} into {lang} text.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, "
    "and write 3 instead of three."
)


class Runtime:
    """The two model servers the app uses, started lazily."""

    def __init__(self):
        self.llm: Server | None = None
        self.embedder: Server | None = None
        self._llm_key: str | None = None
        self._start_lock = threading.Lock()

    def ensure_llm(self) -> Server:
        with self._start_lock:
            s = config.settings()
            if self.llm and self.llm.running() and self._llm_key == s.llm:
                return self.llm
            if self.llm:
                self.llm.stop()
            model = models.local_path(s.llm, "model")
            mmproj = models.local_path(s.llm, "mmproj")
            if model is None or mmproj is None:
                raise RuntimeError(f"{models.CATALOG[s.llm].label} is not downloaded")
            # two slots: live transcription keeps flowing while notes or answers generate
            args = ["-m", str(model), "--mmproj", str(mmproj), "-c", str(s.ctx_size), "-np", "2",
                    "-ngl", "all", "-fa", "auto", "--reasoning", "off", "--jinja"]
            if backend() == "cpu":
                args[args.index("-ngl") + 1] = "0"
            self.llm = Server("gemma", args)
            self.llm.start()
            self._llm_key = s.llm
            return self.llm

    def ensure_embedder(self) -> Server:
        with self._start_lock:
            if self.embedder and self.embedder.running():
                return self.embedder
            model = models.local_path("embeddinggemma-2", "model")
            if model is None:
                raise RuntimeError("EmbeddingGemma 2 is not downloaded")
            # 2048-token context is plenty for ~1,200-character chunks
            args = ["-m", str(model), "--embeddings", "--pooling", "mean", "-c", "8192", "-np", "4",
                    "-ub", "2048", "-b", "2048", "-ngl", "all" if backend() != "cpu" else "0"]
            self.embedder = Server("embed", args)
            self.embedder.start()
            return self.embedder

    def stop(self) -> None:
        for s in (self.llm, self.embedder):
            if s:
                s.stop()

    # --- calls -------------------------------------------------------------

    def transcribe(self, wav: bytes, language: str) -> str:
        srv = self.ensure_llm()
        body = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": ASR_PROMPT.format(lang=language)},
                    {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
                ],
            }],
            "temperature": 0,
            "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = httpx.post(f"{srv.url}/v1/chat/completions", json=body, timeout=120)
        r.raise_for_status()
        return (r.json()["choices"][0]["message"].get("content") or "").strip()

    def chat_stream(self, messages: list[dict], max_tokens: int = 4096, temperature: float = 0.3) -> Iterator[str]:
        srv = self.ensure_llm()
        body = {"messages": messages, "stream": True, "temperature": temperature, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}}
        with httpx.stream("POST", f"{srv.url}/v1/chat/completions", json=body, timeout=httpx.Timeout(30, read=600)) as r:
            if r.status_code >= 400:
                r.read()
                raise RuntimeError(f"llama-server {r.status_code}: {r.text[:500]}")
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta") or {}
                if delta.get("content"):
                    yield delta["content"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        srv = self.ensure_embedder()
        r = httpx.post(f"{srv.url}/v1/embeddings", json={"input": texts}, timeout=300)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


runtime = Runtime()
