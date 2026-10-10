"""SQLite storage for meetings, transcript lines and the knowledge-base index."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from typing import Any

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    status TEXT NOT NULL,          -- recording | done | importing
    notes TEXT NOT NULL DEFAULT '',
    enhanced TEXT NOT NULL DEFAULT '',
    language TEXT NOT NULL DEFAULT 'English'
);
CREATE TABLE IF NOT EXISTS lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    source TEXT NOT NULL,          -- mic | system | file
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS lines_meeting ON lines(meeting_id, start);
CREATE TABLE IF NOT EXISTS kb_files (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    chunks INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS kb_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ref TEXT NOT NULL,             -- file path, or meeting:<id>
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS kb_chunks_ref ON kb_chunks(ref);
"""


class DB:
    def __init__(self, path=None):
        self.conn = sqlite3.connect(path or config.db_path(), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()
        # recover meetings left "recording" by a crash or forced quit
        # (end time from the last transcribed line, so the meeting doesn't show as 0 minutes)
        self.conn.execute("""UPDATE meetings SET status='done', ended_at=COALESCE(ended_at, started_at +
                                 COALESCE((SELECT max(end) FROM lines WHERE lines.meeting_id = meetings.id), 0))
                             WHERE status IN ('recording','importing')""")
        self.conn.commit()

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def _x(self, sql: str, args: tuple = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    # --- meetings ----------------------------------------------------------

    def create_meeting(self, title: str, status: str, language: str) -> str:
        mid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self._x("INSERT INTO meetings(id,title,started_at,status,language) VALUES(?,?,?,?,?)",
                (mid, title, time.time(), status, language))
        return mid

    def meeting(self, mid: str) -> dict[str, Any] | None:
        rows = self._q("SELECT * FROM meetings WHERE id=?", (mid,))
        return dict(rows[0]) if rows else None

    def meetings(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._q("""SELECT m.id, m.title, m.started_at, m.ended_at, m.status,
                                 length(m.notes) AS notes_len, length(m.enhanced) AS enhanced_len,
                                 (SELECT count(*) FROM lines l WHERE l.meeting_id=m.id) AS line_count
                          FROM meetings m ORDER BY m.started_at DESC LIMIT ?""", (limit,))
        return [dict(r) for r in rows]

    def update_meeting(self, mid: str, **fields) -> None:
        allowed = {"title", "ended_at", "status", "notes", "enhanced", "language"}
        bad = set(fields) - allowed
        if bad:
            raise KeyError(", ".join(bad))
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._x(f"UPDATE meetings SET {cols} WHERE id=?", (*fields.values(), mid))

    def delete_meeting(self, mid: str) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM lines WHERE meeting_id=?", (mid,))
            self.conn.execute("DELETE FROM kb_chunks WHERE ref=?", (f"meeting:{mid}",))
            self.conn.execute("DELETE FROM meetings WHERE id=?", (mid,))
            self.conn.commit()

    # --- transcript --------------------------------------------------------

    def add_line(self, mid: str, source: str, start: float, end: float, text: str) -> int:
        return self._x("INSERT INTO lines(meeting_id,source,start,end,text) VALUES(?,?,?,?,?)",
                       (mid, source, start, end, text))

    def delete_line(self, line_id: int) -> None:
        self._x("DELETE FROM lines WHERE id=?", (line_id,))

    def lines(self, mid: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q("SELECT * FROM lines WHERE meeting_id=? ORDER BY start", (mid,))]

    # --- knowledge base ----------------------------------------------------

    def kb_file(self, path: str) -> dict | None:
        rows = self._q("SELECT * FROM kb_files WHERE path=?", (path,))
        return dict(rows[0]) if rows else None

    def kb_has(self, ref: str) -> bool:
        return bool(self._q("SELECT 1 FROM kb_chunks WHERE ref=? LIMIT 1", (ref,)))

    def kb_files(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM kb_files ORDER BY path")]

    def kb_replace(self, ref: str, title: str, chunks: list[str], vecs: np.ndarray,
                   file_meta: tuple[float, int] | None) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM kb_chunks WHERE ref=?", (ref,))
            self.conn.executemany("INSERT INTO kb_chunks(ref,title,text,vec) VALUES(?,?,?,?)",
                                  [(ref, title, t, v.astype(np.float32).tobytes()) for t, v in zip(chunks, vecs)])
            if file_meta is not None:
                self.conn.execute("INSERT OR REPLACE INTO kb_files(path,mtime,size,chunks) VALUES(?,?,?,?)",
                                  (ref, file_meta[0], file_meta[1], len(chunks)))
            self.conn.commit()

    def kb_remove(self, ref: str) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM kb_chunks WHERE ref=?", (ref,))
            self.conn.execute("DELETE FROM kb_files WHERE path=?", (ref,))
            self.conn.commit()

    def kb_all(self) -> tuple[list[dict], np.ndarray]:
        rows = self._q("SELECT id, ref, title, text, vec FROM kb_chunks")
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        meta = [{"id": r["id"], "ref": r["ref"], "title": r["title"], "text": r["text"]} for r in rows]
        mat = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        return meta, mat
