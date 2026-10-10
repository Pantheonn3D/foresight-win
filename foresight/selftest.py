"""Headless end-to-end check used by CI: real llama.cpp binaries, real Gemma
weights, real audio. Prints what it did and returns a process exit code."""

from __future__ import annotations

import re
import time
from pathlib import Path

from . import config, kb, llama, models, notes
from .audio.capture import decode_file
from .audio.segmenter import Segmenter
from .db import DB
from .session import clean, to_wav


def _step(name: str):
    print(f"\n== {name}", flush=True)
    return time.monotonic()


def run(llm: str, wav: str, expect: str, skip_embed: bool, backend: str = "auto") -> int:
    config.update_settings(llm=llm, backend=backend, ctx_size=16384)
    failures = []

    t = _step(f"install llama.cpp {llama.LLAMA_CPP_TAG} ({llama.backend()})")
    if not llama.runtime_installed():
        llama.install_runtime()
    print(f"server: {llama.find_server()}  ({time.monotonic() - t:.0f}s)")

    keys = [llm] + ([] if skip_embed else ["embeddinggemma-2"])
    for key in keys:
        t = _step(f"download {key}")
        entry = models.resolve(key)
        print({p: entry[p]["path"] if entry.get(p) else None for p in ("model", "mmproj")})
        models.download(key)
        print(f"{models.remote_size(key) / 1e9:.2f} GB ({time.monotonic() - t:.0f}s)")

    t = _step("start Gemma server")
    llama.runtime.ensure_llm()
    print(f"ready ({time.monotonic() - t:.0f}s)")
    import sys

    if sys.platform == "win32":
        import ctypes

        in_job = ctypes.c_int(0)
        ctypes.windll.kernel32.IsProcessInJob(ctypes.c_void_p(int(llama.runtime.llm.proc._handle)), None,
                                              ctypes.byref(in_job))
        print(f"llama-server tied to app lifetime (job object): {bool(in_job.value) and llama.job_ok}")
        if not (in_job.value and llama.job_ok):
            failures.append("llama-server is not in the kill-on-close job object")

    t = _step("transcribe")
    audio = decode_file(Path(wav))
    seg = Segmenter()
    segments = seg.feed(audio) + seg.flush()
    print(f"{len(audio) / 16000:.1f}s audio -> {len(segments)} segments")
    lines = []
    for s in segments:
        text = clean(llama.runtime.transcribe(to_wav(s.audio), "English"))
        print(f"  [{s.start:5.1f}-{s.end:5.1f}] {text}")
        if text:
            lines.append({"source": "file", "start": s.start, "end": s.end, "text": text})
    got = set(re.findall(r"\w+", " ".join(l["text"] for l in lines).lower()))
    want = re.findall(r"\w+", expect.lower())
    hit = [w for w in want if w in got]
    recall = len(hit) / max(len(want), 1)
    print(f"expected-word recall {recall:.0%} ({time.monotonic() - t:.0f}s)")
    if recall < 0.8:
        failures.append(f"transcript recall {recall:.0%} < 80%; missing {sorted(set(want) - got)}")

    t = _step("enhance notes")
    out = "".join(notes.enhance("- weather\n- plans", lines, "Self-test")) if lines else ""
    print(out[:1500])
    print(f"({time.monotonic() - t:.0f}s)")
    if len(out.strip()) < 20:
        failures.append("note enhancement returned almost nothing")

    if not skip_embed:
        t = _step("embed + search")
        idx = kb.Index(DB(config.data_dir() / "selftest.db"))
        idx.index_text("a", "budget.txt", "The marketing budget for Q3 is 42,000 euros, approved by Mette on 3 June.")
        idx.index_text("b", "trip.txt", "The team offsite is in Skagen from 12 to 14 September; book the train early.")
        idx.index_text("c", "recipe.txt", "Rugbrød needs a sourdough starter, cracked rye and 20 hours of rest.")
        hits = idx.search("How much money does marketing have next quarter?", k=3)
        for h in hits:
            print(f"  {h['score']:.3f} {h['title']}")
        if not hits or hits[0]["ref"] != "a":
            failures.append("embedding search ranked the wrong document first")
        print(f"({time.monotonic() - t:.0f}s)")

    llama.runtime.stop()
    print("\n== result")
    for f in failures:
        print("FAIL:", f)
    print("PASS" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0
