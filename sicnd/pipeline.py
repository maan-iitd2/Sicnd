"""End-to-end song pipeline: plan -> ACE-Step -> Demucs -> RVC -> mix.

Stages run strictly one at a time and free VRAM between steps so the whole
thing fits in 8 GB.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from . import acestep_engine, config, mixer, narration, rvc_engine, stems
from .llm import SongPlan


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "song"


def run(plan: SongPlan, voice: str | None = None, pitch_shift: int = 0,
        log=print) -> Path:
    """Generate the song described by `plan`. Returns the final MP3 path.

    voice: name of an RVC model in models/rvc/ (without .pth), or None to
    keep ACE-Step's own vocals.
    """
    config.ensure_dirs()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    song_dir = config.OUTPUTS_DIR / f"{stamp}_{_slugify(plan.title)}"
    work = song_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    # 1. ACE-Step: full song (vocals + instrumental in one mix)
    log(f"[1/4] ACE-Step: generating '{plan.title}' "
        f"({plan.duration_sec}s)... this is the slow step.")
    raw_song = acestep_engine.generate_song(plan, work / "acestep_raw.wav")

    current = raw_song

    # 2 + 3. Voice cloning (only if a voice was requested)
    if voice:
        log(f"[2/4] Demucs: splitting vocals from instrumental...")
        vocals, instrumental = stems.split_vocals(raw_song, work / "stems")

        log(f"[3/4] RVC: converting vocals to voice '{voice}'...")
        cloned = rvc_engine.convert_vocals(
            vocals, work / "vocals_cloned.wav", voice, pitch_shift
        )
        current = mixer.remix(instrumental, cloned, work / "remixed.wav")
    else:
        log("[2/4] No voice model requested - keeping ACE-Step vocals.")
        log("[3/4] (skipped RVC)")

    # 3.5 Narration (optional spoken-word intro)
    if plan.narration:
        log("      Kokoro: rendering spoken-word intro...")
        spoken = narration.synthesize(plan.narration, work / "narration.wav")
        current = mixer.prepend_narration(spoken, current, work / "with_intro.wav")

    # 4. Final export
    log("[4/4] ffmpeg: normalizing and exporting MP3...")
    final = mixer.export_mp3(
        Path(current), song_dir / f"{_slugify(plan.title)}.mp3", plan.title
    )

    # Keep lyrics next to the audio for reference
    (song_dir / "lyrics.txt").write_text(
        f"{plan.title}\n\nTags: {plan.style_tags}\n\n{plan.lyrics}\n",
        encoding="utf-8",
    )
    log(f"Done -> {final}")
    return final
