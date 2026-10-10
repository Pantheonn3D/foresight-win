import io
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from foresight import kb, llama, models, server
from foresight.audio.segmenter import SR
from foresight.session import to_wav


@pytest.fixture
def client(fake_runtime, monkeypatch):
    monkeypatch.setattr(llama, "runtime_installed", lambda: True)
    monkeypatch.setattr(models, "is_installed", lambda key: True)
    monkeypatch.setattr(kb, "available", lambda: True)
    monkeypatch.setattr(server.App, "warm", lambda self: None)
    with TestClient(server.create_app(), headers={"x-foresight": "1"}) as c:
        yield c


def speech_wav(seconds=3.0):
    t = np.arange(int(seconds * SR)) / SR
    x = 0.2 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    return to_wav(np.concatenate([np.zeros(SR), x, np.zeros(SR)]).astype(np.float32))


def test_ui_and_state(client):
    assert "Foresight" in client.get("/").text
    assert client.get("/app.js").status_code == 200
    st = client.get("/api/state").json()
    assert st["setup"]["runtime_installed"] is True
    assert st["recording"] is None
    assert "Danish" in st["languages"]


def test_settings_validation(client):
    assert client.post("/api/settings", json={"language": "Danish"}).json()["language"] == "Danish"
    assert client.post("/api/settings", json={"bogus": 1}).status_code == 400


def test_websocket_receives_events(client):
    with client.websocket_connect("/api/events") as ws:
        from foresight.events import bus

        time.sleep(0.05)
        bus.publish("runtime", llm="ready")
        assert json.loads(ws.receive_text()) == {"type": "runtime", "llm": "ready"}


def test_import_enhance_ask_delete(client, fake_runtime):
    _, script = fake_runtime
    script["transcripts"] = ["We ship on Thursday."]
    r = client.post("/api/import", files={"file": ("standup.wav", io.BytesIO(speech_wav()), "audio/wav")})
    assert r.status_code == 200
    mid = r.json()["meeting_id"]
    for _ in range(100):
        m = client.get(f"/api/meetings/{mid}").json()
        if m["status"] == "done":
            break
        time.sleep(0.05)
    assert m["title"] == "standup"
    assert [l["text"] for l in m["lines"]] == ["We ship on Thursday."]

    assert client.patch(f"/api/meetings/{mid}", json={"notes": "- ship date"}).status_code == 200
    script["chat"] = "## Ship date\n- Thursday"
    out = client.post(f"/api/meetings/{mid}/enhance").text
    assert "Thursday" in out
    assert client.get(f"/api/meetings/{mid}").json()["enhanced"].startswith("## Ship date")

    script["chat"] = "Thursday [1]."
    lines = client.post("/api/ask", json={"question": "When do we ship?", "meeting_id": mid}).text.splitlines()
    first = json.loads(lines[0])
    assert first["sources"][0]["ref"] == f"meeting:{mid}"
    assert "".join(json.loads(l).get("t", "") for l in lines[1:]).strip() == "Thursday [1]."

    assert client.delete(f"/api/meetings/{mid}").status_code == 200
    assert client.get(f"/api/meetings/{mid}").status_code == 404


def test_enhance_without_transcript_is_refused(client):
    from foresight import db as dbmod

    app_db = dbmod.DB()
    mid = app_db.create_meeting("empty", "done", "English")
    assert client.post(f"/api/meetings/{mid}/enhance").status_code == 400


def test_open_path_is_limited_to_linked_folders(client, tmp_path):
    secret = tmp_path / "elsewhere.txt"
    secret.write_text("x")
    assert client.post("/api/open", json={"path": str(secret)}).status_code == 403


def test_cross_site_requests_are_refused(client):
    # a page on another site can send a simple POST, but not one with a custom header
    r = client.post("/api/recording/start", headers={"x-foresight": ""})
    assert r.status_code == 403
    assert client.get("/api/state", headers={"host": "evil.example"}).status_code == 400
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/events", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_text()


def test_delete_refused_while_transcribing(client, fake_runtime):
    from foresight import db as dbmod, session
    from foresight.audio.segmenter import Segment

    _, script = fake_runtime
    gate = __import__("threading").Event()
    orig = llama.runtime.transcribe
    llama.runtime.transcribe = lambda wav, lang: (gate.wait(5), "hello there")[1]
    try:
        d = dbmod.DB()
        mid = d.create_meeting("busy", "done", "English")
        t = session.Transcriber(d, mid, "English", dedupe_echo=False)
        t.submit("file", Segment(0, 1, np.zeros(SR, dtype=np.float32)))
        assert client.delete(f"/api/meetings/{mid}").status_code == 409
        gate.set()
        t.finish(5)
        assert client.delete(f"/api/meetings/{mid}").status_code == 200
    finally:
        llama.runtime.transcribe = orig
