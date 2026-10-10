"""Turns the user's shorthand notes plus the transcript into full meeting notes."""

from __future__ import annotations

from typing import Iterator

from . import config
from .llama import runtime

# Rough chars-per-token for mixed English/Danish speech; only used to decide
# whether the transcript needs condensing before it fits a context slot.
CHARS_PER_TOKEN = 3.5
DIRECT_LIMIT_TOKENS = 20000
CONDENSE_CHUNK_CHARS = 40000


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def transcript_text(lines: list[dict]) -> str:
    name = config.settings().your_name or "You"
    who = {"mic": name, "system": "Others", "file": "Speaker"}
    return "\n".join(f"[{fmt_time(l['start'])}] {who.get(l['source'], l['source'])}: {l['text']}" for l in lines)


SYSTEM = (
    "You write meeting notes. You are given the user's own rough notes and a transcript. "
    "The transcript labels the user's microphone as '{name}' and everyone heard through the computer as 'Others'. "
    "Rules:\n"
    "- Keep every one of the user's own lines, in their order and wording, as the skeleton. Expand each with the "
    "concrete details from the transcript that belong to it: numbers, names, dates, reasons, who said what.\n"
    "- Add things the user did not note only if they matter: decisions, commitments, open questions, deadlines.\n"
    "- Never invent facts. If the transcript is unclear about something, say it is unclear.\n"
    "- End with '## Action items' as '- [ ] owner: task (due date if stated)' only if there are any.\n"
    "- Output Markdown only. No preamble, no closing remarks."
)


def _condense(transcript: str) -> str:
    """Summarise long transcripts chunk by chunk so the final pass fits one context slot."""
    parts = []
    for i in range(0, len(transcript), CONDENSE_CHUNK_CHARS):
        chunk = transcript[i:i + CONDENSE_CHUNK_CHARS]
        msgs = [{"role": "system", "content": "Condense this meeting transcript section into dense, factual "
                                              "bullet points with timestamps. Keep every number, name, date, "
                                              "decision and commitment. No commentary."},
                {"role": "user", "content": chunk}]
        parts.append("".join(runtime.chat_stream(msgs, max_tokens=2048, temperature=0.1)))
    return "\n".join(parts)


def enhance(notes: str, lines: list[dict], title: str) -> Iterator[str]:
    transcript = transcript_text(lines)
    if not transcript.strip():
        raise ValueError("There is no transcript yet, so there is nothing to expand the notes with.")
    if len(transcript) / CHARS_PER_TOKEN > DIRECT_LIMIT_TOKENS:
        transcript = "(Condensed from a long transcript)\n" + _condense(transcript)
    name = config.settings().your_name or "You"
    user = (f"Meeting: {title}\n\n## My notes\n{notes.strip() or '(none — write the notes from the transcript)'}"
            f"\n\n## Transcript\n{transcript}")
    msgs = [{"role": "system", "content": SYSTEM.format(name=name)}, {"role": "user", "content": user}]
    yield from runtime.chat_stream(msgs, max_tokens=4096, temperature=0.2)
