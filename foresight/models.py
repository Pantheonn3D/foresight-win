"""Model catalog and downloader.

Files are resolved against the Hugging Face tree API once, then pinned in
models/manifest.json so the app never needs the network again after setup.
Downloads resume from .part files and are verified against the LFS sha256.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from . import config

log = logging.getLogger(__name__)

HF = "https://huggingface.co"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo: str
    # fnmatch patterns tried in order; the first pattern that matches a file wins
    patterns: tuple[str, ...]
    mmproj_patterns: tuple[str, ...] = ()
    role: str = "llm"  # "llm" or "embed"


CATALOG: dict[str, ModelSpec] = {
    "gemma-4-e4b": ModelSpec(
        key="gemma-4-e4b",
        label="Gemma 4 E4B",
        repo="ggml-org/gemma-4-E4B-it-GGUF",
        patterns=("*Q8_0.gguf", "*q8_0.gguf", "*Q4_K_M.gguf", "*.gguf"),
        mmproj_patterns=("mmproj*f16*.gguf", "mmproj*F16*.gguf", "mmproj*bf16*.gguf", "mmproj*.gguf"),
    ),
    "gemma-4-e2b": ModelSpec(
        key="gemma-4-e2b",
        label="Gemma 4 E2B",
        repo="ggml-org/gemma-4-E2B-it-GGUF",
        patterns=("*Q8_0.gguf", "*q8_0.gguf", "*Q4_K_M.gguf", "*.gguf"),
        mmproj_patterns=("mmproj*f16*.gguf", "mmproj*F16*.gguf", "mmproj*bf16*.gguf", "mmproj*.gguf"),
    ),
    "embeddinggemma-2": ModelSpec(
        key="embeddinggemma-2",
        label="EmbeddingGemma 2",
        repo="ggml-org/embeddinggemma-2-GGUF",
        patterns=("*Q8_0.gguf", "*q8_0.gguf", "*F16.gguf", "*f16.gguf", "*.gguf"),
        role="embed",
    ),
}


@dataclass
class RemoteFile:
    path: str
    size: int
    sha256: str | None


def _http() -> httpx.Client:
    headers = {"User-Agent": "foresight-win/0.1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(headers=headers, follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0))


def list_repo(repo: str) -> list[RemoteFile]:
    with _http() as c:
        r = c.get(f"{HF}/api/models/{repo}/tree/main")
        r.raise_for_status()
    out = []
    for item in r.json():
        if item.get("type") != "file":
            continue
        lfs = item.get("lfs") or {}
        out.append(RemoteFile(item["path"], int(lfs.get("size") or item.get("size") or 0), lfs.get("oid")))
    return out


def pick(files: list[RemoteFile], patterns: tuple[str, ...], exclude_mmproj: bool) -> RemoteFile | None:
    for pat in patterns:
        for f in files:
            name = f.path.rsplit("/", 1)[-1]
            if exclude_mmproj and name.lower().startswith("mmproj"):
                continue
            if fnmatch.fnmatchcase(name, pat):
                return f
    return None


def _manifest_path() -> Path:
    return config.models_dir() / "manifest.json"


def load_manifest() -> dict:
    p = _manifest_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("manifest.json is corrupt; re-resolving models")
    return {}


def _save_manifest(m: dict) -> None:
    p = _manifest_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=2), encoding="utf-8")
    tmp.replace(p)


def resolve(key: str) -> dict:
    """Return {"model": {...}, "mmproj": {...}|None} for a catalog key, querying HF if needed."""
    manifest = load_manifest()
    if key in manifest:
        return manifest[key]
    spec = CATALOG[key]
    files = list_repo(spec.repo)
    main = pick(files, spec.patterns, exclude_mmproj=True)
    if main is None:
        raise RuntimeError(f"No GGUF file found in {spec.repo}")
    entry = {"repo": spec.repo, "model": vars(main), "mmproj": None}
    if spec.mmproj_patterns:
        mm = pick(files, spec.mmproj_patterns, exclude_mmproj=False)
        if mm is None:
            raise RuntimeError(f"No mmproj (audio encoder) file found in {spec.repo}")
        entry["mmproj"] = vars(mm)
    manifest[key] = entry
    _save_manifest(manifest)
    return entry


def local_path(key: str, part: str = "model") -> Path | None:
    entry = load_manifest().get(key)
    if not entry or not entry.get(part):
        return None
    p = config.models_dir() / key / Path(entry[part]["path"]).name
    return p if p.exists() else None


def is_installed(key: str) -> bool:
    entry = load_manifest().get(key)
    if not entry:
        return False
    parts = ["model"] + (["mmproj"] if entry.get("mmproj") else [])
    return all(local_path(key, p) is not None for p in parts)


ProgressFn = Callable[[int, int], None]


def download_file(url: str, dest: Path, size: int, sha256: str | None,
                  progress: ProgressFn | None = None, cancel: threading.Event | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if size and have > size:
        part.unlink()
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with _http() as c, c.stream("GET", url, headers=headers) as r:
        if r.status_code == 416:  # already complete
            pass
        else:
            r.raise_for_status()
            if have and r.status_code != 206:
                have = 0  # server ignored Range; start over
            total = size or (have + int(r.headers.get("content-length", 0)))
            with open(part, "ab" if have else "wb") as f:
                done = have
                for chunk in r.iter_bytes(1 << 20):
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError("download cancelled")
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
    if sha256:
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        if h.hexdigest() != sha256:
            part.unlink()
            raise RuntimeError(f"Checksum mismatch for {dest.name}; the partial file was removed, retry the download")
    os.replace(part, dest)


def download(key: str, progress: ProgressFn | None = None, cancel: threading.Event | None = None) -> None:
    entry = resolve(key)
    parts = [p for p in ("model", "mmproj") if entry.get(p)]
    total = sum(entry[p]["size"] for p in parts)
    offset = 0
    for p in parts:
        f = entry[p]
        dest = config.models_dir() / key / Path(f["path"]).name
        if not dest.exists():
            base = offset
            download_file(f"{HF}/{entry['repo']}/resolve/main/{f['path']}", dest, f["size"], f["sha256"],
                          (lambda d, t, base=base: progress(base + d, total)) if progress else None, cancel)
        offset += f["size"]
        if progress:
            progress(offset, total)


def remote_size(key: str) -> int | None:
    entry = load_manifest().get(key)
    if not entry:
        return None
    return sum(entry[p]["size"] for p in ("model", "mmproj") if entry.get(p))
