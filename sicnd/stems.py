"""Vocal/instrumental stem separation via Demucs.

ACE-Step outputs a fully mixed song. To apply a cloned voice we split it
into vocals + instrumental, convert only the vocals with RVC, then remix.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import config


def split_vocals(song_path: Path, work_dir: Path) -> tuple[Path, Path]:
    """Return (vocals_wav, instrumental_wav) split out of `song_path`."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", config.DEMUCS_MODEL,
        "-o", str(work_dir),
        str(song_path),
    ]
    subprocess.run(cmd, check=True)

    stem_dir = work_dir / config.DEMUCS_MODEL / song_path.stem
    vocals = stem_dir / "vocals.wav"
    instrumental = stem_dir / "no_vocals.wav"
    if not vocals.exists() or not instrumental.exists():
        raise RuntimeError(f"Demucs did not produce stems in {stem_dir}")
    return vocals, instrumental
