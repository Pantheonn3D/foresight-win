"""UI development server with seeded meetings and fake models (no GPU, no audio).

    python scripts/demo_server.py [--setup] [--recording] [--port 47830]

--setup shows the first-run download screen; --recording shows a live meeting.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser()
ap.add_argument("--setup", action="store_true")
ap.add_argument("--recording", action="store_true")
ap.add_argument("--port", type=int, default=47830)
args = ap.parse_args()

os.environ["FORESIGHT_HOME"] = tempfile.mkdtemp(prefix="foresight-demo-")

import uvicorn  # noqa: E402

from foresight import config, kb, llama, models, server  # noqa: E402

installed = not args.setup
llama.runtime_installed = lambda: installed
llama.backend = lambda: "cuda"
models.is_installed = lambda key: installed
models.remote_size = lambda key: {"gemma-4-e4b": 8_960_000_000, "gemma-4-e2b": 5_210_000_000,
                                  "embeddinggemma-2": 1_180_000_000}[key]
models.resolve = lambda key: {}
llama.runtime_download_size = lambda: 1_037_000_000
kb.available = lambda: installed


def fake_stream(messages, **kw):
    text = ("The release moves to Thursday 15 October [1]. Mette approved the 42,000 kr budget, "
            "but only if the usability test runs before the release [1][2].")
    for w in text.split(" "):
        time.sleep(0.02)
        yield w + " "


llama.runtime.chat_stream = fake_stream
llama.runtime.embed = lambda texts: [[1.0] + [0.0] * 15 for _ in texts]


def ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def seed(app: server.App) -> None:
    db = app.db
    config.update_settings(language="Danish", your_name="Jens", folders=[r"C:\Users\jens\Documents\AAU\P5"])
    m1 = db.create_meeting("P5 supervisor meeting", "done", "Danish")
    db.update_meeting(m1, notes="- test plan?\n- ethics form\n- Lars: recruit participants",
                      enhanced=("## Test plan?\n- Supervisor wants the think-aloud test moved before the "
                                "prototype freeze, so findings can still change the design.\n"
                                "- 6 participants, 45 minutes each, in the Create lab.\n\n"
                                "## Ethics form\n- Must be submitted by Friday 16 October; the faculty "
                                "replies within 5 working days.\n\n"
                                "## Lars: recruit participants\n- Lars recruits from 3rd-semester Medialogy; "
                                "aims for 3 people who have never used a voice assistant.\n\n"
                                "## Action items\n- [ ] Jens: draft the test script (Wednesday 14 October)\n"
                                "- [ ] Lars: send recruitment message (Monday 12 October)\n"
                                "- [ ] Sofie: submit the ethics form (Friday 16 October)"))
    with db.lock:
        db.conn.execute("UPDATE meetings SET started_at=?, ended_at=? WHERE id=?",
                        (ts("2026-10-08T14:02:00"), ts("2026-10-08T14:49:00"), m1))
        db.conn.commit()
    lines = [
        ("system", 4, "Okay, shall we start with the test plan? I read the draft yesterday."),
        ("mic", 11, "Yes. We planned the think-aloud test for week 46, after the prototype freeze."),
        ("system", 19, "That's the part I'd change. If you test after the freeze, nothing you learn can go into the design."),
        ("system", 29, "Move it before the freeze, even with a rougher prototype."),
        ("mic", 37, "Then we need participants by the 19th. Lars, can you recruit?"),
        ("system", 44, "Sure, I'll ask third semester Medialogy. Six people, forty-five minutes each?"),
        ("mic", 53, "Six is fine. Try to get three who have never used a voice assistant."),
        ("system", 62, "And the ethics form has to be in by Friday the 16th. The faculty takes about five working days."),
        ("mic", 71, "Sofie said she'd handle the form. I'll write the test script by Wednesday."),
    ]
    for src, start, text in lines:
        db.add_line(m1, src, start, start + 6, text)
    m2 = db.create_meeting("Interview: Danske Spil UX team", "done", "English")
    with db.lock:
        db.conn.execute("UPDATE meetings SET started_at=?, ended_at=? WHERE id=?",
                        (ts("2026-10-06T10:30:00"), ts("2026-10-06T11:12:00"), m2))
        db.conn.commit()
    db.add_line(m2, "system", 3, 8, "Thanks for joining. Could you start by telling us how you onboard new players?")
    m3 = db.create_meeting("Group standup", "done", "Danish")
    with db.lock:
        db.conn.execute("UPDATE meetings SET started_at=?, ended_at=? WHERE id=?",
                        (time.time() - 3600 * 3, time.time() - 3600 * 3 + 14 * 60, m3))
        db.conn.commit()
    db.add_line(m3, "mic", 2, 6, "Quick round: what did everyone finish yesterday?")

    if args.recording:
        live = db.create_meeting("Thesis kickoff with Mette", "recording", "Danish")
        db.update_meeting(live, notes="- scope: human-AI handoff\n- deadline?")
        for src, start, text in [
            ("mic", 3, "Thanks for taking the time. I want to scope the project around handoffs between people and AI agents."),
            ("system", 12, "Good. What do you mean by handoff, concretely?"),
            ("mic", 18, "The moment an agent stops and needs a decision. Most interfaces hide why it stopped."),
            ("system", 27, "Then your first deliverable is a taxonomy of those stops. Have it by the 30th of October."),
        ]:
            db.add_line(live, src, start, start + 5, text)

        class FakeRecorder:
            paused = False
            t = 0.0

            def levels(self):
                FakeRecorder.t += 0.1
                import math
                return {"mic": round(0.25 + 0.2 * math.sin(self.t * 3), 3), "system": 0.04}

        class FakeRecording:
            meeting_id = live
            started = time.monotonic() - 754
            recorder = FakeRecorder()
            paused = False

            def set_paused(self, v):
                self.paused = v

            def stop(self):
                db.update_meeting(live, status="done", ended_at=time.time())

        app.recording = FakeRecording()
    app.runtime_state.update(llm="ready" if installed else "stopped", embed="ready" if installed else "stopped")
    app.warm = lambda: None
    app.index.stats = lambda: {"files": 214, "chunks": 1932, "indexing": False}


api = server.create_app()
seed(api.state.core)
uvicorn.run(api, host="127.0.0.1", port=args.port, log_level="warning")
