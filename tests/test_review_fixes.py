import time

import httpx
import numpy as np
import pytest

from foresight import config, llama, notes
from foresight.audio import capture
from foresight.db import DB


class FakeServer:
    def __init__(self, fail):
        self.fail = fail
        self.starts = 0

    def start(self):
        self.starts += 1
        if self.fail:
            raise RuntimeError("CUDA out of memory")


def test_failed_start_backs_off_and_reports_state():
    rt = llama.Runtime()
    seen = []
    rt.on_state = lambda kind, state, error=None: seen.append((kind, state, error))
    srv = FakeServer(fail=True)
    with pytest.raises(RuntimeError, match="out of memory"):
        rt._start("llm", srv)
    # a second segment right after must not trigger another 30-second model load
    with pytest.raises(RuntimeError, match="out of memory"):
        rt._start("llm", srv)
    assert srv.starts == 1
    assert seen == [("llm", "starting", None), ("llm", "error", "CUDA out of memory")]
    rt._failed["llm"] = (time.monotonic() - llama.RETRY_START_AFTER - 1, "old")
    srv.fail = False
    rt._start("llm", srv)
    assert srv.starts == 2 and seen[-1] == ("llm", "ready", None)


def test_stream_error_payload_is_readable(monkeypatch):
    rt = llama.Runtime()

    class Srv:
        url = "http://x"

    monkeypatch.setattr(rt, "ensure_llm", lambda: Srv())
    body = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: {"error":{"code":400,"message":"context too long"}}\n\n'
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=body))
    client = httpx.Client(transport=transport)
    monkeypatch.setattr(httpx, "stream", client.stream)
    out = []
    with pytest.raises(RuntimeError, match="context too long"):
        for tok in rt.chat_stream([{"role": "user", "content": "x"}]):
            out.append(tok)
    assert out == ["Hi"]


def test_cjk_transcripts_are_condensed_much_earlier(fake_runtime):
    calls, _ = fake_runtime
    # ~30k characters: fine in Danish (~8.6k tokens), far over the direct limit in Japanese
    lines = [{"source": "system", "start": i, "end": i + 1, "text": "あ" * 300} for i in range(100)]
    "".join(notes.enhance("", lines, "t", "Danish"))
    assert len(calls["chat"]) == 1
    calls["chat"].clear()
    "".join(notes.enhance("", lines, "t", "Japanese"))
    assert len(calls["chat"]) > 1


def test_device_lookup_by_name_survives_renumbering(monkeypatch):
    import sys
    import types

    fake = types.SimpleNamespace(paWASAPI=13)
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", fake)

    class PA:
        devs = [
            {"name": "Speakers (Realtek)", "hostApi": 2, "maxInputChannels": 0, "maxOutputChannels": 2},
            {"name": "Jabra Evolve2 65", "hostApi": 2, "maxInputChannels": 1, "maxOutputChannels": 0},
            {"name": "Jabra Evolve2 65", "hostApi": 0, "maxInputChannels": 1, "maxOutputChannels": 0},  # MME twin
        ]

        def get_host_api_info_by_type(self, t):
            return {"index": 2}

        def get_device_count(self):
            return len(self.devs)

        def get_device_info_by_index(self, i):
            return {**self.devs[i], "index": i}

    assert capture._find(PA(), "Jabra Evolve2 65", "input")["index"] == 1
    assert capture._find(PA(), "Jabra Evolve2 65", "output") is None  # wrong kind -> default
    assert capture._find(PA(), "Unplugged headset", "input") is None
    assert capture._find(PA(), None, "input") is None


def test_old_index_settings_reset_to_default(home):
    (home / "settings.json").write_text('{"mic_device": 3, "speaker_device": "Speakers", "language": "Danish"}')
    s = config.Settings.load()
    assert (s.mic_device, s.speaker_device, s.language) == (None, "Speakers", "Danish")


def test_crash_recovery_keeps_meeting_duration():
    db = DB()
    mid = db.create_meeting("crashed", "recording", "English")
    db.add_line(mid, "mic", 0, 5, "first")
    db.add_line(mid, "system", 1200, 1206, "last words")
    db.conn.close()
    m = DB().meeting(mid)
    assert m["status"] == "done"
    assert m["ended_at"] - m["started_at"] == pytest.approx(1206)


def test_meter_decays_during_loopback_silence():
    t = capture.Track("system", 16000, 1, t0=time.monotonic(), on_segment=lambda n, s: None)
    t.level = 0.8
    t.last_audio = time.monotonic() - 1.0
    assert t.meter() < 0.01
    t.last_audio = time.monotonic()
    assert t.meter() == pytest.approx(0.8)
    t.q.put(None)
    t._thread.join()
