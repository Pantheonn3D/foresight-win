import hashlib
import re

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FORESIGHT_HOME", str(tmp_path))
    from foresight import config

    monkeypatch.setattr(config, "_settings", None)
    return tmp_path


def fake_vec(text: str, dim: int = 64) -> list[float]:
    """Bag-of-words hashing embedding: shared words -> similar vectors."""
    v = np.zeros(dim)
    for w in re.findall(r"\w+", text.lower().split("|", 1)[-1]):
        if len(w) < 3:
            continue
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    return list(v)


@pytest.fixture
def fake_runtime(monkeypatch):
    from foresight import llama

    calls = {"transcribe": [], "chat": []}
    script = {"transcripts": [], "chat": "## Notes\n- expanded"}

    def transcribe(wav, language):
        calls["transcribe"].append(language)
        return script["transcripts"].pop(0) if script["transcripts"] else ""

    def chat_stream(messages, max_tokens=4096, temperature=0.3):
        calls["chat"].append(messages)
        for word in script["chat"].split(" "):
            yield word + " "

    monkeypatch.setattr(llama.runtime, "transcribe", transcribe)
    monkeypatch.setattr(llama.runtime, "chat_stream", chat_stream)
    monkeypatch.setattr(llama.runtime, "embed", lambda texts: [fake_vec(t) for t in texts])
    return calls, script
