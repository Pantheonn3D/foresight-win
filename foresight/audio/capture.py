"""Microphone + system-audio capture on Windows via WASAPI (PyAudioWPatch).

Each source becomes a Track: raw device audio -> mono -> 16 kHz -> segmenter.
WASAPI loopback delivers nothing while the system is silent, so tracks fill
gaps with silence against the wall clock to keep both channels aligned with
the meeting timeline.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soxr

from .segmenter import SR, Segment, Segmenter

log = logging.getLogger(__name__)

SegmentFn = Callable[[str, Segment], None]


@dataclass
class Device:
    index: int
    name: str
    kind: str  # "input" or "output"
    default: bool


def list_devices() -> list[Device]:
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return []
    out: list[Device] = []
    with pyaudio.PyAudio() as p:
        try:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            return []
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["hostApi"] != wasapi["index"] or d.get("isLoopbackDevice"):
                continue
            if d["maxInputChannels"] > 0:
                out.append(Device(i, d["name"], "input", i == wasapi["defaultInputDevice"]))
            if d["maxOutputChannels"] > 0:
                out.append(Device(i, d["name"], "output", i == wasapi["defaultOutputDevice"]))
    return out


class Track:
    """Turns device callbacks into 16 kHz segments. Independent of PortAudio so it can be tested."""

    def __init__(self, name: str, rate: int, channels: int, t0: float, on_segment: SegmentFn,
                 save_to: Path | None = None, fill_gaps: bool = False):
        self.name = name
        self.channels = channels
        self.t0 = t0
        self.on_segment = on_segment
        self.fill_gaps = fill_gaps
        self.resampler = soxr.ResampleStream(rate, SR, 1, dtype="float32", quality="HQ")
        self.segmenter = Segmenter()
        self.written = 0  # 16 kHz samples emitted so far
        self.level = 0.0  # recent peak, 0..1, for the UI meter
        self.q: queue.Queue[tuple[float, bytes] | None] = queue.Queue()
        self._writer = None
        if save_to is not None:
            import soundfile as sf
            self._writer = sf.SoundFile(save_to, "w", samplerate=SR, channels=1, format="FLAC", subtype="PCM_16")
        self._thread = threading.Thread(target=self._run, name=f"track-{name}", daemon=True)
        self._thread.start()

    def push(self, raw: bytes, now: float | None = None) -> None:
        self.q.put((time.monotonic() if now is None else now, raw))

    def close(self) -> None:
        self.q.put(None)
        self._thread.join(timeout=30)

    def _emit(self, mono16: np.ndarray) -> None:
        if not len(mono16):
            return
        if self._writer is not None:
            self._writer.write(mono16)
        self.written += len(mono16)
        for seg in self.segmenter.feed(mono16):
            self.on_segment(self.name, seg)

    def process(self, when: float, raw: bytes) -> None:
        x = np.frombuffer(raw, dtype=np.float32)
        if self.channels > 1:
            x = x.reshape(-1, self.channels).mean(axis=1)
        self.level = max(float(np.abs(x).max(initial=0.0)), self.level * 0.6)
        y = self.resampler.resample_chunk(x.astype(np.float32, copy=False))
        if self.fill_gaps:
            expected = int((when - self.t0) * SR) - len(y)
            gap = expected - self.written
            if gap > int(0.1 * SR):
                self._emit(np.zeros(gap, dtype=np.float32))
        self._emit(y)

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            try:
                self.process(*item)
            except Exception:
                log.exception("audio processing failed on %s", self.name)
        try:
            tail = self.resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
            self._emit(tail)
            for seg in self.segmenter.flush():
                self.on_segment(self.name, seg)
        finally:
            if self._writer is not None:
                self._writer.close()


class Recorder:
    """Owns the PortAudio streams for one recording session."""

    def __init__(self, on_segment: SegmentFn, folder: Path | None, mic: int | None, speaker: int | None,
                 system_audio: bool):
        import pyaudiowpatch as pyaudio

        self.pa = pyaudio.PyAudio()
        self.tracks: dict[str, Track] = {}
        self.streams = []
        self.t0 = time.monotonic()
        self.paused = False
        try:
            mic_info = (self.pa.get_device_info_by_index(mic) if mic is not None
                        else self.pa.get_default_wasapi_device(d_in=True))
            # both tracks fill gaps so pauses and dropped callbacks keep them on one timeline
            self._open("mic", mic_info, on_segment, folder, fill_gaps=True)
            if system_audio:
                out_info = (self.pa.get_device_info_by_index(speaker) if speaker is not None
                            else self.pa.get_default_wasapi_device(d_out=True))
                loop_info = self.pa.get_wasapi_loopback_analogue_by_dict(out_info)
                self._open("system", loop_info, on_segment, folder, fill_gaps=True)
        except Exception:
            self.stop()
            raise

    def _open(self, name: str, info: dict, on_segment: SegmentFn, folder: Path | None, fill_gaps: bool) -> None:
        import pyaudiowpatch as pyaudio

        rate = int(info["defaultSampleRate"])
        channels = max(1, int(info["maxInputChannels"]))
        track = Track(name, rate, channels, self.t0, on_segment,
                      save_to=(folder / f"{name}.flac") if folder else None, fill_gaps=fill_gaps)
        self.tracks[name] = track

        def callback(data, frames, time_info, status):
            if not self.paused:
                track.push(data)
            return (None, pyaudio.paContinue)

        stream = self.pa.open(format=pyaudio.paFloat32, channels=channels, rate=rate, input=True,
                              input_device_index=info["index"], frames_per_buffer=int(rate * 0.05),
                              stream_callback=callback)
        self.streams.append(stream)
        log.info("capturing %s from %r at %d Hz x%d", name, info["name"], rate, channels)

    def levels(self) -> dict[str, float]:
        return {k: round(t.level, 3) for k, t in self.tracks.items()}

    def stop(self) -> None:
        for s in self.streams:
            try:
                s.stop_stream()
                s.close()
            except Exception:
                log.exception("closing stream")
        self.streams = []
        for t in self.tracks.values():
            t.close()
        self.pa.terminate()


def decode_file(path: Path) -> np.ndarray:
    """Any audio/video file -> float32 mono 16 kHz, via PyAV's bundled FFmpeg."""
    import av

    chunks = []
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("file has no audio track")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)
