"""HTTP + WebSocket API consumed by the web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, kb, llama, models, notes, session
from .db import DB
from .events import bus

log = logging.getLogger(__name__)
WEB = Path(__file__).parent / "web"


class App:
    def __init__(self):
        self.db = DB()
        self.index = kb.Index(self.db)
        self.recording: session.Recording | None = None
        self.rec_lock = threading.Lock()
        self.setup_cancel: threading.Event | None = None
        self.setup_state: dict = {"running": False}
        self.runtime_state = {"llm": "stopped", "embed": "stopped", "error": None}

    # --- runtime warm-up ---------------------------------------------------

    def warm(self) -> None:
        def work():
            s = config.settings()
            if llama.runtime_installed() and models.is_installed(s.llm):
                self._set_runtime(llm="starting")
                try:
                    llama.runtime.ensure_llm()
                    self._set_runtime(llm="ready")
                except Exception as e:
                    log.exception("llm start failed")
                    self._set_runtime(llm="error", error=str(e))
            if llama.runtime_installed() and kb.available():
                self._set_runtime(embed="starting")
                try:
                    llama.runtime.ensure_embedder()
                    self._set_runtime(embed="ready")
                    self.index.scan()
                except Exception as e:
                    log.exception("embedder start failed")
                    self._set_runtime(embed="error", error=str(e))

        threading.Thread(target=work, name="warm", daemon=True).start()

    def _set_runtime(self, **kw) -> None:
        self.runtime_state.update(kw)
        bus.publish("runtime", **self.runtime_state)

    # --- setup -------------------------------------------------------------

    def start_setup(self, keys: list[str]) -> None:
        if self.setup_state.get("running"):
            raise HTTPException(409, "Setup is already running")
        bad = [k for k in keys if k not in models.CATALOG]
        if bad:
            raise HTTPException(400, f"Unknown models: {', '.join(bad)}")
        self.setup_cancel = threading.Event()
        cancel = self.setup_cancel
        steps = ([("runtime", f"llama.cpp {llama.LLAMA_CPP_TAG} ({llama.backend().upper()})")]
                 if not llama.runtime_installed() else [])
        steps += [(k, models.CATALOG[k].label) for k in keys if not models.is_installed(k)]

        def report(i, key, label, done, total):
            self.setup_state = {"running": True, "step": i + 1, "steps": len(steps), "key": key, "label": label,
                                "done": done, "total": total}
            bus.publish("setup", **self.setup_state)

        def work():
            try:
                for i, (key, label) in enumerate(steps):
                    report(i, key, label, 0, 0)
                    cb = lambda d, t, i=i, key=key, label=label: report(i, key, label, d, t)
                    if key == "runtime":
                        llama.install_runtime(cb, cancel)
                    else:
                        models.download(key, cb, cancel)
                self.setup_state = {"running": False, "finished": True}
                bus.publish("setup", **self.setup_state)
                self.warm()
            except Exception as e:
                log.exception("setup failed")
                msg = "Cancelled" if isinstance(e, InterruptedError) else str(e)
                self.setup_state = {"running": False, "error": msg}
                bus.publish("setup", **self.setup_state)

        self.setup_state = {"running": True, "step": 0, "steps": len(steps)}
        threading.Thread(target=work, name="setup", daemon=True).start()

    def state(self) -> dict:
        s = config.settings()
        rec = None
        if self.recording:
            rec = {"meeting_id": self.recording.meeting_id, "paused": self.recording.paused,
                   "elapsed": time.monotonic() - self.recording.started}
        return {
            "setup": {
                "runtime_installed": llama.runtime_installed(),
                "backend": llama.backend(),
                "llama_tag": llama.LLAMA_CPP_TAG,
                "models": [{"key": k, "label": m.label, "role": m.role, "installed": models.is_installed(k),
                            "size": models.remote_size(k)} for k, m in models.CATALOG.items()],
                "progress": self.setup_state,
            },
            "runtime": self.runtime_state,
            "recording": rec,
            "settings": asdict(s),
            "languages": config.LANGUAGES,
            "kb": self.index.stats(),
            "data_dir": str(config.data_dir()),
        }


def create_app() -> FastAPI:
    app = App()

    @asynccontextmanager
    async def lifespan(_api: FastAPI):
        app.warm()

        async def levels():
            while True:
                await asyncio.sleep(0.1)
                r = app.recording
                if r is not None:
                    bus.publish("levels", levels=r.recorder.levels(),
                                elapsed=round(time.monotonic() - r.started, 1))

        task = asyncio.create_task(levels())
        yield
        task.cancel()
        if app.recording:
            await asyncio.to_thread(app.recording.stop)
        llama.runtime.stop()

    api = FastAPI(title="Foresight", lifespan=lifespan)
    api.state.core = app

    @api.get("/api/state")
    def get_state():
        return app.state()

    @api.post("/api/setup/install")
    def setup_install(keys: list[str] = Body(..., embed=True)):
        app.start_setup(keys)
        return {"ok": True}

    @api.get("/api/setup/sizes")
    def setup_sizes():
        out: dict = {"free": shutil.disk_usage(config.data_dir()).free}
        for key in models.CATALOG:
            try:
                models.resolve(key)
                out[key] = models.remote_size(key)
            except Exception as e:
                log.warning("could not resolve %s: %s", key, e)
                out[key] = None
        out["runtime"] = None if llama.runtime_installed() else llama.runtime_download_size()
        return out

    @api.post("/api/setup/cancel")
    def setup_cancel():
        if app.setup_cancel:
            app.setup_cancel.set()
        return {"ok": True}

    @api.get("/api/devices")
    def devices():
        from .audio.capture import list_devices

        return [asdict(d) for d in list_devices()]

    @api.post("/api/settings")
    def set_settings(changes: dict = Body(...)):
        old = config.settings()
        try:
            new = config.update_settings(**changes)
        except (KeyError, TypeError) as e:
            raise HTTPException(400, str(e))
        if new.llm != old.llm or new.ctx_size != old.ctx_size or new.backend != old.backend:
            if llama.runtime.llm:
                llama.runtime.llm.stop()
            app._set_runtime(llm="stopped")
            app.warm()
        if new.folders != old.folders:
            threading.Thread(target=app.index.scan, daemon=True).start()
        return asdict(new)

    # --- meetings ----------------------------------------------------------

    @api.get("/api/meetings")
    def list_meetings():
        return app.db.meetings()

    @api.get("/api/meetings/{mid}")
    def get_meeting(mid: str):
        m = app.db.meeting(mid)
        if not m:
            raise HTTPException(404, "No such meeting")
        folder = config.meetings_dir() / mid
        audio = [p.stem for p in folder.glob("*.flac")] if folder.exists() else []
        return {**m, "lines": app.db.lines(mid), "audio": audio}

    @api.get("/api/meetings/{mid}/audio/{track}")
    def get_audio(mid: str, track: str):
        if track not in ("mic", "system"):
            raise HTTPException(404)
        p = config.meetings_dir() / mid / f"{track}.flac"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="audio/flac")

    @api.patch("/api/meetings/{mid}")
    def patch_meeting(mid: str, changes: dict = Body(...)):
        allowed = {k: v for k, v in changes.items() if k in ("title", "notes", "enhanced")}
        if not app.db.meeting(mid):
            raise HTTPException(404)
        app.db.update_meeting(mid, **allowed)
        return {"ok": True}

    @api.delete("/api/meetings/{mid}")
    def delete_meeting(mid: str):
        if app.recording and app.recording.meeting_id == mid:
            raise HTTPException(409, "Stop the recording first")
        app.db.delete_meeting(mid)
        shutil.rmtree(config.meetings_dir() / mid, ignore_errors=True)
        return {"ok": True}

    @api.post("/api/meetings/{mid}/enhance")
    def enhance(mid: str):
        m = app.db.meeting(mid)
        if not m:
            raise HTTPException(404)
        lines = app.db.lines(mid)
        if not lines:
            raise HTTPException(400, "There is no transcript yet, so there is nothing to expand the notes with.")

        def gen():
            out = []
            try:
                for tok in notes.enhance(m["notes"], lines, m["title"]):
                    out.append(tok)
                    yield tok
            except Exception as e:
                log.exception("enhance failed")
                yield f"\n\n**Stopped: {e}**"
                return
            app.db.update_meeting(mid, enhanced="".join(out).strip())
            if m["status"] == "done":
                threading.Thread(target=app.index.index_meeting, args=(mid,), daemon=True).start()

        return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

    # --- recording ---------------------------------------------------------

    @api.post("/api/recording/start")
    def rec_start(title: str | None = Body(None, embed=True)):
        with app.rec_lock:
            if app.recording:
                raise HTTPException(409, "Already recording")
            try:
                app.recording = session.Recording(app.db, title)
            except Exception as e:
                log.exception("could not start recording")
                raise HTTPException(500, f"Could not open audio devices: {e}")
            mid = app.recording.meeting_id
        bus.publish("recording", meeting_id=mid, state="recording")
        return {"meeting_id": mid}

    @api.post("/api/recording/pause")
    def rec_pause(paused: bool = Body(..., embed=True)):
        if not app.recording:
            raise HTTPException(409, "Not recording")
        app.recording.set_paused(paused)
        bus.publish("recording", meeting_id=app.recording.meeting_id, state="paused" if paused else "recording")
        return {"ok": True}

    @api.post("/api/recording/stop")
    def rec_stop():
        with app.rec_lock:
            r = app.recording
            if not r:
                raise HTTPException(409, "Not recording")
            app.recording = None

        def finish():
            r.stop()
            app.index.index_meeting(r.meeting_id)

        threading.Thread(target=finish, name="stop", daemon=True).start()
        return {"meeting_id": r.meeting_id}

    @api.post("/api/import")
    async def import_audio(file: UploadFile):
        suffix = Path(file.filename or "audio").suffix
        tmpdir = Path(tempfile.mkdtemp(prefix="foresight-"))
        dest = tmpdir / (Path(file.filename or "audio").stem + suffix)
        with open(dest, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        mid = session.import_file(app.db, dest, cleanup=True)
        return {"meeting_id": mid}

    # --- knowledge base ----------------------------------------------------

    @api.post("/api/kb/scan")
    def kb_scan():
        threading.Thread(target=app.index.scan, daemon=True).start()
        return {"ok": True}

    @api.get("/api/kb/files")
    def kb_files():
        return app.db.kb_files()

    @api.post("/api/ask")
    def ask(question: str = Body(...), meeting_id: str | None = Body(None)):
        if not question.strip():
            raise HTTPException(400, "Empty question")
        try:
            sources, stream = kb.ask(app.index, app.db, question, meeting_id)
        except (RuntimeError, KeyError) as e:
            raise HTTPException(400, str(e))

        def gen():
            yield json.dumps({"sources": sources}) + "\n"
            try:
                for tok in stream:
                    yield json.dumps({"t": tok}) + "\n"
            except Exception as e:
                log.exception("ask failed")
                yield json.dumps({"error": str(e)}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @api.post("/api/open")
    def open_path(path: str = Body(..., embed=True)):
        """Reveal a source file in Explorer. Only paths inside linked folders are allowed."""
        p = Path(path).resolve()
        roots = [Path(f).resolve() for f in config.settings().folders]
        if not any(p.is_relative_to(r) for r in roots) or not p.exists():
            raise HTTPException(403, "Not inside a linked folder")
        import os
        import sys

        if sys.platform == "win32":
            os.startfile(p)  # noqa: S606 - user-initiated open of their own file
        return {"ok": True}

    # --- events ------------------------------------------------------------

    @api.websocket("/api/events")
    async def events(ws: WebSocket):
        await ws.accept()
        q = bus.subscribe()
        try:
            while True:
                msg = await q.get()
                await ws.send_text(json.dumps(msg))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            bus.unsubscribe(q)

    api.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return api
