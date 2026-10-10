import numpy as np
import pytest

from foresight.audio.capture import Track
from foresight.audio.segmenter import MAX_SECONDS, SR, Segmenter

rng = np.random.default_rng(0)


def noise(seconds, db):
    amp = 10 ** (db / 20)
    return (rng.standard_normal(int(seconds * SR)) * amp).astype(np.float32)


def voice(seconds, db=-20):
    # amplitude-modulated harmonic tone: loud, speech-like envelope
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate((180, 360, 540, 1100)))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)
    x = x * env
    x = x / np.sqrt(np.mean(x ** 2)) * 10 ** (db / 20)
    return x.astype(np.float32)


def run(audio, chunk=1600):
    s = Segmenter()
    out = []
    for i in range(0, len(audio), chunk):
        out += s.feed(audio[i:i + chunk])
    return out + s.flush()


def test_two_utterances_split_on_pause():
    audio = np.concatenate([noise(1, -55), voice(2), noise(1.2, -55), voice(3), noise(1, -55)])
    segs = run(audio)
    assert len(segs) == 2
    assert segs[0].start == pytest.approx(1.0, abs=0.35)
    assert segs[0].end == pytest.approx(3.0, abs=0.35)
    assert segs[1].start == pytest.approx(4.2, abs=0.35)
    assert segs[1].end - segs[1].start == pytest.approx(3.0, abs=0.5)
    for s in segs:
        assert len(s.audio) == pytest.approx((s.end - s.start) * SR, abs=1)


def test_silence_and_room_noise_produce_nothing():
    assert run(noise(10, -70)) == []
    assert run(noise(10, -45)) == []  # steady fan noise above the absolute floor


def test_click_is_ignored():
    audio = np.concatenate([noise(1, -60), voice(0.1), noise(2, -60)])
    assert run(audio) == []


def test_monologue_is_cut_under_model_limit_without_losing_audio():
    # 70 s of speech with short 0.25 s breaths every 6 s: no pause long enough to end a segment early
    parts = []
    for _ in range(12):
        parts += [voice(5.75), noise(0.25, -60)]
    audio = np.concatenate(parts)[: 70 * SR]
    segs = run(audio)
    assert len(segs) >= 3
    assert all(s.end - s.start <= MAX_SECONDS + 0.01 for s in segs)
    covered = sum(s.end - s.start for s in segs)
    assert covered == pytest.approx(70, abs=2.5)
    # segments are contiguous-ish and ordered
    assert all(b.start >= a.start for a, b in zip(segs, segs[1:]))


def test_quiet_speaker_in_quiet_room_detected():
    audio = np.concatenate([noise(2, -65), voice(2, db=-42), noise(2, -65)])
    assert len(run(audio)) == 1


def test_track_fills_loopback_gaps_and_keeps_timeline():
    segs = []
    t = Track("system", 48000, 2, t0=100.0, on_segment=lambda n, s: segs.append(s), fill_gaps=True)
    v = voice(1.0)
    stereo48 = np.repeat(np.interp(np.arange(48000) / 48000, np.arange(SR) / SR, v), 2).astype(np.float32)
    # sound arrives at t=101 (1 s after start), nothing before it: loopback was silent
    t.process(101.0 + 1.0, stereo48.tobytes())
    silence = np.zeros(48000 * 2, dtype=np.float32)
    t.process(103.0, silence.tobytes())
    t.q.put(None)
    t._thread.join()
    assert t.written == pytest.approx(3 * SR, abs=SR * 0.15)
    assert len(segs) == 1
    assert segs[0].start == pytest.approx(1.0, abs=0.4)


def test_noise_that_starts_mid_meeting_stops_counting_as_speech():
    # air conditioning kicks in at -38 dB: at most one short false segment, then speech still found
    audio = np.concatenate([noise(3, -65), noise(20, -38), voice(2, db=-18) + noise(2, -38), noise(3, -38)])
    segs = run(audio)
    speech = [s for s in segs if s.start > 22]
    assert len(speech) == 1
    assert sum(s.end - s.start for s in segs if s.start < 22) < 12
