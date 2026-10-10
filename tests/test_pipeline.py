import io
import wave

import numpy as np
import pytest

from foresight import kb, models, notes
from foresight.audio.segmenter import SR, Segment
from foresight.db import DB
from foresight.session import Transcriber, clean, similar, to_wav


def seg(start, end):
    return Segment(start, end, np.zeros(int((end - start) * SR), dtype=np.float32))


@pytest.mark.parametrize("raw,want", [
    ("  Hello   there. ", "Hello there."),
    ("[music]", ""),
    ("(silence)", ""),
    ("...", ""),
    ("*inaudible*", ""),
    ("the the the the the the the plan", "the plan"),
    ("we need to go go now", "we need to go go now"),
    ("Budget is 1.7 million", "Budget is 1.7 million"),
])
def test_clean(raw, want):
    assert clean(raw) == want


def test_similar_handles_partial_echo():
    assert similar("so the deadline is Friday", "Okay so the deadline is Friday for the draft") > 0.9
    assert similar("I'll send the invoice", "the weather in Aalborg is grey") < 0.6


def test_wav_roundtrip():
    x = np.sin(np.linspace(0, 100, SR)).astype(np.float32) * 0.5
    with wave.open(io.BytesIO(to_wav(x))) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (SR, 1, 2, SR)


def test_transcriber_drops_mic_echo_in_both_orders(fake_runtime):
    calls, script = fake_runtime
    db = DB()
    mid = db.create_meeting("t", "recording", "Danish")
    t = Transcriber(db, mid, "Danish", dedupe_echo=True)
    script["transcripts"] = [
        "So the release moves to Thursday",               # system first
        "the release moves to Thursday",                  # mic hears speakers -> dropped
        "I disagree, Thursday is too early",               # mic, real speech -> kept
        "we should invite Lars as well",                  # mic first (echo transcribed early)
        "We should invite Lars as well, he owns billing",  # system later -> removes the mic echo
    ]
    t.submit("system", seg(0, 3))
    t.submit("mic", seg(0.2, 3.1))
    t.submit("mic", seg(4, 6))
    t.submit("mic", seg(10, 12))
    t.submit("system", seg(9.8, 13))
    t.finish()
    lines = db.lines(mid)
    assert [(l["source"], l["text"]) for l in lines] == [
        ("system", "So the release moves to Thursday"),
        ("mic", "I disagree, Thursday is too early"),
        ("system", "We should invite Lars as well, he owns billing"),
    ]
    assert calls["transcribe"] == ["Danish"] * 5


def test_transcriber_keeps_everything_with_headphones(fake_runtime):
    _, script = fake_runtime
    db = DB()
    mid = db.create_meeting("t", "recording", "English")
    t = Transcriber(db, mid, "English", dedupe_echo=False)
    script["transcripts"] = ["same words here", "same words here", ""]
    for s in ("system", "mic", "mic"):
        t.submit(s, seg(0, 2))
    t.finish()
    assert len(db.lines(mid)) == 2  # the empty transcript is skipped


def test_chunk_sizes_and_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(40))
    chunks = kb.chunk(text)
    assert all(len(c) <= kb.CHUNK_CHARS + 5 for c in chunks)
    assert len(chunks) > 5
    # every paragraph marker survives somewhere
    joined = " ".join(chunks)
    assert all(f"Paragraph {i}." in joined for i in range(40))
    # consecutive chunks share text
    assert chunks[1].split(" … ")[0].strip() in chunks[0]


def test_chunk_splits_giant_paragraph():
    chunks = kb.chunk("x" * 5000)
    assert len(chunks) >= 5 and all(len(c) <= kb.CHUNK_CHARS + 5 for c in chunks)


def test_search_and_ask(fake_runtime, monkeypatch, tmp_path):
    calls, script = fake_runtime
    monkeypatch.setattr(kb, "available", lambda: True)
    db = DB()
    idx = kb.Index(db)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "budget.md").write_text("Marketing budget for Q3 is 42000 euros approved by Mette.")
    (tmp_path / "docs" / "trip.txt").write_text("Offsite in Skagen in September, book trains early.")
    (tmp_path / "docs" / "skip.exe").write_bytes(b"MZ")
    from foresight import config

    config.update_settings(folders=[str(tmp_path / "docs")])
    idx.scan()
    assert idx.stats()["files"] == 2
    hits = idx.search("marketing budget euros")
    assert hits[0]["title"] == "budget.md"
    script["chat"] = "It is 42000 euros [1]."
    sources, stream = kb.ask(idx, db, "What is the marketing budget?")
    assert sources[0]["title"] == "budget.md"
    assert "".join(stream).strip() == "It is 42000 euros [1]."
    prompt = calls["chat"][-1][1]["content"]
    assert "[1] budget.md" in prompt
    # removed files leave the index
    (tmp_path / "docs" / "trip.txt").unlink()
    idx.scan()
    assert idx.stats()["files"] == 1


def test_enhance_prompt_contains_notes_and_labelled_transcript(fake_runtime):
    calls, script = fake_runtime
    lines = [{"source": "mic", "start": 5, "end": 7, "text": "Can we ship Friday?"},
             {"source": "system", "start": 65, "end": 68, "text": "Only if QA signs off by Thursday."}]
    out = "".join(notes.enhance("- ship date", lines, "Release sync"))
    assert "expanded" in out
    user = calls["chat"][-1][1]["content"]
    assert "- ship date" in user
    assert "[00:05] You: Can we ship Friday?" in user
    assert "[01:05] Others: Only if QA signs off by Thursday." in user


def test_enhance_condenses_long_transcripts(fake_runtime):
    calls, _ = fake_runtime
    lines = [{"source": "system", "start": i * 10, "end": i * 10 + 9, "text": "word " * 80} for i in range(300)]
    "".join(notes.enhance("", lines, "Long"))
    assert len(calls["chat"]) > 2  # condense passes, then the final pass


def test_model_pick_prefers_q8_and_mmproj():
    files = [models.RemoteFile(p, 1, None) for p in
             ("README.md", "gemma-4-E4B-it-Q4_K_M.gguf", "gemma-4-E4B-it-Q8_0.gguf",
              "mmproj-gemma-4-E4B-it-f16.gguf", "mmproj-gemma-4-E4B-it-Q8_0.gguf")]
    spec = models.CATALOG["gemma-4-e4b"]
    assert models.pick(files, spec.patterns, True).path == "gemma-4-E4B-it-Q8_0.gguf"
    assert models.pick(files, spec.mmproj_patterns, False).path == "mmproj-gemma-4-E4B-it-f16.gguf"
