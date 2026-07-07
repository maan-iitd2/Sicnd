"""Ollama-powered songwriting brain.

Turns a conversational request ("make me a sad indie song like Anuv Jain")
into a structured SongPlan: title, ACE-Step style tags, structured lyrics,
duration, and optional spoken-word narration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import ollama

from . import config

SYSTEM_PROMPT = """\
You are a professional songwriter and music producer working inside a local
AI music generation tool. The user describes the song they want in plain
language. You reply with ONLY a JSON object (no markdown, no commentary)
with exactly these keys:

{
  "title": "short song title",
  "style_tags": "comma-separated tags for a music generation model: genre, \
mood, instruments, vocal type, tempo. Example: 'indie folk, acoustic guitar, \
melancholic, soft male vocals, intimate, fingerpicking, 80 bpm, lo-fi warmth'",
  "lyrics": "full lyrics using structure markers [verse], [chorus], [bridge], \
[outro] on their own lines. Write in the language the user asks for \
(support Hindi/Hinglish if they want an Anuv Jain style). Keep it singable: \
short lines, clear rhymes.",
  "duration_sec": 150,
  "narration": ""
}

Rules:
- style_tags must describe the MUSIC, never name real artists. If the user
  names an artist, translate that into genre/mood/instrument tags instead
  (e.g. 'like Anuv Jain' -> 'indie folk, soft hindi male vocals, acoustic
  guitar, dreamy, heartfelt, minimal production').
- duration_sec: integer between 60 and 240. Default 150 unless asked.
- narration: a short spoken-word intro/outro ONLY if the user asks for
  spoken word, otherwise "".
- If the user is refining a previous song, update the previous plan rather
  than starting from scratch.
"""


@dataclass
class SongPlan:
    title: str
    style_tags: str
    lyrics: str
    duration_sec: int = config.DEFAULT_DURATION_SEC
    narration: str = ""
    raw: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Title      : {self.title}",
            f"Style tags : {self.style_tags}",
            f"Duration   : {self.duration_sec}s",
        ]
        if self.narration:
            lines.append(f"Narration  : {self.narration[:120]}")
        lines.append("Lyrics     :")
        lines.append(self.lyrics)
        return "\n".join(lines)


class Songwriter:
    """Stateful chat wrapper so follow-up messages refine the current plan."""

    def __init__(self, model: str = config.OLLAMA_MODEL):
        self.model = model
        self.client = ollama.Client(host=config.OLLAMA_HOST)
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def write(self, user_message: str) -> SongPlan:
        self.history.append({"role": "user", "content": user_message})
        response = self.client.chat(
            model=self.model,
            messages=self.history,
            format="json",
            options={"temperature": 0.8},
        )
        content = response["message"]["content"]
        self.history.append({"role": "assistant", "content": content})
        return self._parse(content)

    def unload(self) -> None:
        """Free the LLM from VRAM before ACE-Step needs it (keep_alive=0)."""
        try:
            self.client.generate(model=self.model, prompt="", keep_alive=0)
        except Exception:
            pass  # best effort; Ollama also evicts on its own timeout

    @staticmethod
    def _parse(content: str) -> SongPlan:
        data = json.loads(content)
        duration = int(data.get("duration_sec") or config.DEFAULT_DURATION_SEC)
        duration = max(60, min(duration, 240))
        return SongPlan(
            title=str(data.get("title") or "untitled").strip(),
            style_tags=str(data.get("style_tags") or "").strip(),
            lyrics=str(data.get("lyrics") or "").strip(),
            duration_sec=duration,
            narration=str(data.get("narration") or "").strip(),
            raw=data,
        )
