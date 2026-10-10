"""Energy-based voice activity segmentation.

Splits a 16 kHz mono float stream into utterances that fit Gemma 4's 30-second
audio window. A segment closes after a pause, or is forced closed at
MAX_SECONDS, preferring the quietest recent frame as the cut point so words
aren't sliced in half.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

SR = 16000
FRAME = 480  # 30 ms
MAX_SECONDS = 28.0
MIN_SPEECH_SECONDS = 0.35
PRE_ROLL = 0.3
PAUSE_SECONDS = 0.7  # pause that ends an utterance
SHORT_PAUSE_SECONDS = 0.3  # once a segment is long, accept a shorter pause
LONG_SEGMENT_SECONDS = 15.0
ABS_FLOOR_DB = -60.0  # below this nothing is speech, however quiet the room
MARGIN_DB = 10.0  # speech must sit this far above the tracked noise floor
FLOOR_WINDOW = int(8 * SR / FRAME)  # noise floor = quiet percentile of the last 8 s
FLOOR_PERCENTILE = 10
DEFAULT_FLOOR_DB = -50.0


@dataclass
class Segment:
    start: float  # seconds from stream start
    end: float
    audio: np.ndarray  # float32 mono 16 kHz


def _db(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-10)
    return 20.0 * np.log10(rms)


class Segmenter:
    def __init__(self):
        self.noise_db = DEFAULT_FLOOR_DB
        self._levels: deque[float] = deque(maxlen=FLOOR_WINDOW)
        self._pending = np.zeros(0, dtype=np.float32)
        self._t = 0  # samples consumed into frames
        self._buf: list[np.ndarray] = []  # frames of the open segment
        self._buf_db: list[float] = []
        self._start = 0  # sample index where the open segment starts
        self._speech_frames = 0
        self._silence_run = 0
        self._preroll: list[np.ndarray] = []
        self._active = False

    def _threshold(self) -> float:
        return max(ABS_FLOOR_DB, self.noise_db + MARGIN_DB)

    def feed(self, samples: np.ndarray) -> list[Segment]:
        out: list[Segment] = []
        data = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        n = len(data) // FRAME
        self._pending = data[n * FRAME:]
        for i in range(n):
            frame = data[i * FRAME:(i + 1) * FRAME]
            seg = self._frame(frame)
            if seg is not None:
                out.append(seg)
        return out

    def _frame(self, frame: np.ndarray) -> Segment | None:
        level = _db(frame)
        self._levels.append(level)
        # Minimum statistics: even continuous speech has quiet gaps between words, so a
        # low percentile over 8 s tracks the room's noise without creeping up under talk.
        if self._t % (10 * FRAME) == 0 or len(self._levels) < 34:
            pct = float(np.percentile(self._levels, FLOOR_PERCENTILE))
            # with under a second of history the percentile may be all speech; don't trust it upward
            self.noise_db = min(pct if len(self._levels) >= 34 else min(pct, DEFAULT_FLOOR_DB), -25.0)
        speech = level > self._threshold()

        result = None
        if not self._active:
            self._preroll.append(frame)
            if len(self._preroll) > int(PRE_ROLL * SR / FRAME):
                self._preroll.pop(0)
            if speech:
                self._active = True
                self._buf = list(self._preroll)
                self._buf_db = [ABS_FLOOR_DB] * (len(self._preroll) - 1) + [level]
                self._start = self._t - (len(self._preroll) - 1) * FRAME
                self._preroll = []
                self._speech_frames = 1
                self._silence_run = 0
        else:
            self._buf.append(frame)
            self._buf_db.append(level)
            if speech:
                self._speech_frames += 1
                self._silence_run = 0
            else:
                self._silence_run += 1
            dur = len(self._buf) * FRAME / SR
            pause = self._silence_run * FRAME / SR
            limit = SHORT_PAUSE_SECONDS if dur > LONG_SEGMENT_SECONDS else PAUSE_SECONDS
            if pause >= limit:
                # keep a little trailing silence; drop the rest
                keep = len(self._buf) - self._silence_run + int(0.2 * SR / FRAME)
                result = self._close(min(keep, len(self._buf)))
            elif dur >= MAX_SECONDS:
                result = self._force_cut()
        self._t += FRAME
        return result

    def _close(self, nframes: int) -> Segment | None:
        frames = self._buf[:nframes]
        speech = self._speech_frames
        start = self._start
        self._active = False
        self._buf, self._buf_db = [], []
        self._preroll = []
        self._speech_frames = 0
        self._silence_run = 0
        if speech * FRAME / SR < MIN_SPEECH_SECONDS:
            return None
        audio = np.concatenate(frames)
        return Segment(start / SR, (start + len(audio)) / SR, audio)

    def _force_cut(self) -> Segment | None:
        # cut at the quietest frame in the last 8 seconds; carry the remainder over
        window = int(8 * SR / FRAME)
        lo = max(1, len(self._buf_db) - window)
        cut = lo + int(np.argmin(self._buf_db[lo:]))
        head, tail, tail_db = self._buf[:cut], self._buf[cut:], self._buf_db[cut:]
        start = self._start
        speech = self._speech_frames
        self._buf, self._buf_db = tail, tail_db
        self._start = start + cut * FRAME
        self._speech_frames = sum(1 for d in tail_db if d > self._threshold())
        self._silence_run = 0
        if speech * FRAME / SR < MIN_SPEECH_SECONDS or not head:
            return None
        audio = np.concatenate(head)
        return Segment(start / SR, (start + len(audio)) / SR, audio)

    def flush(self) -> list[Segment]:
        if not self._active:
            return []
        seg = self._close(len(self._buf))
        return [seg] if seg else []
