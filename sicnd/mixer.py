"""ffmpeg-based mixing and export.

Two jobs:
1. remix converted vocals back over the instrumental stem
2. optionally prepend a narration clip, then export a loudness-normalized MP3
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import config


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (Windows: "
            "`winget install Gyan.FFmpeg`) and restart the terminal."
        )
    return exe


def _run(args: list[str]) -> None:
    subprocess.run(
        [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True,
    )


def remix(instrumental: Path, vocals: Path, out_path: Path) -> Path:
    """Overlay vocals on the instrumental without renormalizing levels."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "-i", str(instrumental),
        "-i", str(vocals),
        "-filter_complex",
        (
            f"[0:a]aformat=sample_rates={config.FINAL_SAMPLE_RATE}:"
            f"channel_layouts=stereo[i];"
            f"[1:a]aformat=sample_rates={config.FINAL_SAMPLE_RATE}:"
            f"channel_layouts=stereo[v];"
            f"[i][v]amix=inputs=2:duration=longest:normalize=0[out]"
        ),
        "-map", "[out]",
        str(out_path),
    ])
    return out_path


def prepend_narration(narration: Path, song: Path, out_path: Path) -> Path:
    """Narration first, then the song, with a short crossfade."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sr = config.FINAL_SAMPLE_RATE
    _run([
        "-i", str(narration),
        "-i", str(song),
        "-filter_complex",
        (
            f"[0:a]aformat=sample_rates={sr}:channel_layouts=stereo[n];"
            f"[1:a]aformat=sample_rates={sr}:channel_layouts=stereo[s];"
            f"[n][s]acrossfade=d=1.5[out]"
        ),
        "-map", "[out]",
        str(out_path),
    ])
    return out_path


def export_mp3(wav_path: Path, mp3_path: Path, title: str = "") -> Path:
    """Loudness-normalize and encode the final MP3."""
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-i", str(wav_path),
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-codec:a", "libmp3lame",
        "-q:a", config.MP3_QUALITY,
    ]
    if title:
        args += ["-metadata", f"title={title}"]
    args.append(str(mp3_path))
    _run(args)
    return mp3_path
