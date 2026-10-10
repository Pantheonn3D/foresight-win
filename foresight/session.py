"""Recording sessions and file imports: audio segments in, transcript lines out."""

from __future__ import annotations

import io
import logging
import queue
import re
import threading
import time
import wave
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from . import config
from .audio.segmenter import SR, Segment, Segmenter
from .db import DB
from .events import bus
from .llama import runtime

log = logging.getLogger(__name__)

ECHO_WINDOW = 2.5  # seconds of slack when matching a mic line to a system line
ECHO_SIMILARITY = 0.6


def to_wav(audio: np.ndarray) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


_BRACKETED = re.compile(r"^\s*[\[\(\*].*[\]\)\*]\s*$")
_NON_SPEECH = {"", ".", "...", "…", "-", "no speech", "silence", "inaudible", "music"}


def clean(text: str) -> str:
    text = " ".join(text.split())
    if _BRACKETED.match(text) or text.lower().strip(" .[]()*") in _NON_SPEECH:
        return ""
    # collapse runaway repetition ("the the the the ...") that LLM ASR produces on noise
    text = re.sub(r"\b(\w+(?:\s+\w+){0,3}?)(?:\s+\1\b){4,}", r"\1", text, flags=re.IGNORECASE)
    return text.strip()


def _norm(t: str) -> str:
    return " ".join(re.findall(r"\w+", t.lower()))


def similar(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    # mic often catches only part of what the speakers played
    short, long_ = sorted((a, b), key=len)
    m = SequenceMatcher(None, short, long_, autojunk=False)
    matched = sum(block.size for block in m.get_matching_blocks())
    return matched / len(short)


class Transcriber:
    """Serialises segments through the model and writes transcript lines."""

    def __init__(self, db: DB, meeting_id: str, language: str, dedupe_echo: bool):
        self.db = db
        self.meeting_id = meeting_id
        self.language = language
        self.dedupe_echo = dedupe_echo
        self.q: queue.Queue[tuple[str, Segment] | None] = queue.Queue()
        self.pending = 0
        self._count_lock = threading.Lock()
        self._lines: list[dict] = []
        self._thread = threading.Thread(target=self._run, name="transcriber", daemon=True)
        self._thread.start()

    def submit(self, source: str, seg: Segment) -> None:
        with self._count_lock:
            self.pending += 1
            pending = self.pending
        self.q.put((source, seg))
        bus.publish("backlog", meeting_id=self.meeting_id, pending=pending)

    def finish(self, timeout: float | None = None) -> None:
        self.q.put(None)
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            source, seg = item
            try:
                self._handle(source, seg)
            except Exception as e:
                log.exception("transcription failed")
                bus.publish("error", meeting_id=self.meeting_id, where="transcription", message=str(e))
            finally:
                with self._count_lock:
                    self.pending -= 1
                    pending = self.pending
                bus.publish("backlog", meeting_id=self.meeting_id, pending=pending)

    def _handle(self, source: str, seg: Segment) -> None:
        text = clean(runtime.transcribe(to_wav(seg.audio), self.language))
        if not text:
            return
        line = {"source": source, "start": round(seg.start, 2), "end": round(seg.end, 2), "text": text}
        if self.dedupe_echo and source == "mic":
            if any(self._is_echo(line, o) for o in self._lines if o["source"] == "system"):
                log.info("dropped mic echo: %s", text)
                return
        if self.dedupe_echo and source == "system":
            for o in [o for o in self._lines if o["source"] == "mic" and self._is_echo(o, line)]:
                self.db.delete_line(o["id"])
                self._lines.remove(o)
                bus.publish("line_removed", meeting_id=self.meeting_id, id=o["id"])
        line["id"] = self.db.add_line(self.meeting_id, source, line["start"], line["end"], text)
        self._lines.append(line)
        bus.publish("line", meeting_id=self.meeting_id, line=line)

    @staticmethod
    def _is_echo(mic: dict, system: dict) -> bool:
        overlap = min(mic["end"], system["end"]) - max(mic["start"], system["start"])
        if overlap < -ECHO_WINDOW:
            return False
        return similar(mic["text"], system["text"]) >= ECHO_SIMILARITY


class Recording:
    def __init__(self, db: DB, title: str | None = None):
        from .audio.capture import Recorder

        s = config.settings()
        self.db = db
        self.meeting_id = db.create_meeting(title or time.strftime("Meeting %d %b %H:%M"), "recording", s.language)
        folder = config.meetings_dir() / self.meeting_id
        folder.mkdir(parents=True, exist_ok=True)
        self.transcriber = Transcriber(db, self.meeting_id, s.language,
                                       dedupe_echo=s.capture_system_audio and not s.headphones)
        self.started = time.monotonic()
        try:
            self.recorder = Recorder(self.transcriber.submit, folder if s.save_audio else None,
                                     s.mic_device, s.speaker_device, s.capture_system_audio)
        except Exception:
            self.transcriber.finish()
            db.delete_meeting(self.meeting_id)
            raise

    @property
    def paused(self) -> bool:
        return self.recorder.paused

    def set_paused(self, value: bool) -> None:
        self.recorder.paused = value

    def stop(self) -> None:
        self.recorder.stop()  # flushes the last segments into the transcriber
        self.db.update_meeting(self.meeting_id, ended_at=time.time())
        bus.publish("recording", meeting_id=self.meeting_id, state="finishing")
        self.transcriber.finish()
        self.db.update_meeting(self.meeting_id, status="done")
        bus.publish("recording", meeting_id=self.meeting_id, state="done")


def import_file(db: DB, path: Path, cleanup: bool = False) -> str:
    """Transcribe an audio/video file into a new meeting. Returns the meeting id; work continues in the background."""
    from .audio.capture import decode_file

    s = config.settings()
    mid = db.create_meeting(path.stem, "importing", s.language)

    def work():
        try:
            audio = decode_file(path)
            if cleanup:
                import shutil

                shutil.rmtree(path.parent, ignore_errors=True)
            seg = Segmenter()
            t = Transcriber(db, mid, s.language, dedupe_echo=False)
            step = SR * 10
            for i in range(0, len(audio), step):
                for sgm in seg.feed(audio[i:i + step]):
                    t.submit("file", sgm)
            for sgm in seg.flush():
                t.submit("file", sgm)
            t.finish()
            db.update_meeting(mid, status="done", ended_at=time.time())
            bus.publish("recording", meeting_id=mid, state="done")
        except Exception as e:
            log.exception("import failed")
            db.update_meeting(mid, status="done", ended_at=time.time())
            bus.publish("error", meeting_id=mid, where="import", message=str(e))

    threading.Thread(target=work, name="import", daemon=True).start()
    return mid
