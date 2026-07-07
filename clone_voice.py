#!/usr/bin/env python
"""Prepare an RVC voice-cloning dataset from reference tracks.

Drop full songs (mp3/wav/flac/m4a) of the target artist into
reference_tracks/<voice_name>/ and run:

    python clone_voice.py anuv_jain

This will:
  1. isolate the vocals of every track with Demucs (instruments removed)
  2. slice the vocals into clean ~8s training segments, dropping silence
  3. write the dataset to datasets/<voice_name>/

Training the actual RVC model is a heavier, interactive job that's best done
once in Applio (a free, local RVC trainer with a simple UI):

  1. git clone https://github.com/IAHispano/Applio  (or grab a release zip)
  2. run its run-install.bat, then run-applio.bat
  3. Train tab -> dataset folder: datasets/<voice_name>/
     - sample rate 40k, ~200-300 epochs for ~10 min of clean vocals
  4. Copy the resulting .pth and .index files to:
        models/rvc/<voice_name>.pth
        models/rvc/<voice_name>.index

After that, `python main.py --voice <voice_name>` uses the cloned voice.

Only clone voices you have the right to use (your own, or with the artist's
permission, for private/personal experiments).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sicnd import config

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
SEGMENT_SECONDS = 8


def find_tracks(voice_name: str) -> list[Path]:
    src = config.REFERENCE_DIR / voice_name
    if not src.is_dir():
        sys.exit(
            f"No folder {src}.\nCreate it and drop a few songs of the artist "
            "inside, e.g. reference_tracks/anuv_jain/baarishein.mp3"
        )
    tracks = sorted(p for p in src.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not tracks:
        sys.exit(f"No audio files found in {src}")
    return tracks


def isolate_vocals(tracks: list[Path], work_dir: Path) -> list[Path]:
    """Run Demucs two-stem separation on every reference track."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", config.DEMUCS_MODEL,
        "-o", str(work_dir),
        *[str(t) for t in tracks],
    ]
    subprocess.run(cmd, check=True)
    vocals = [
        work_dir / config.DEMUCS_MODEL / t.stem / "vocals.wav" for t in tracks
    ]
    missing = [v for v in vocals if not v.exists()]
    if missing:
        sys.exit(f"Demucs did not produce: {missing}")
    return vocals


def slice_segments(vocals: list[Path], dataset_dir: Path) -> int:
    """Trim silence and cut vocals into fixed-length training segments."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, vocal in enumerate(vocals):
        pattern = dataset_dir / f"seg_{i:02d}_%04d.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(vocal),
                # drop everything quieter than -35 dB for >0.7s (gaps between
                # vocal lines), then mono 40 kHz which RVC training expects
                "-af",
                (
                    "silenceremove=start_periods=1:start_threshold=-35dB:"
                    "stop_periods=-1:stop_threshold=-35dB:stop_duration=0.7,"
                    "aformat=sample_rates=40000:channel_layouts=mono"
                ),
                "-f", "segment", "-segment_time", str(SEGMENT_SECONDS),
                str(pattern),
            ],
            check=True,
        )
        count += len(list(dataset_dir.glob(f"seg_{i:02d}_*.wav")))
    # Segments under ~2s hurt training; drop them.
    for seg in list(dataset_dir.glob("seg_*.wav")):
        if seg.stat().st_size < 2 * 40000 * 2:  # ~2s of 16-bit mono 40kHz
            seg.unlink()
            count -= 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build an RVC training dataset from reference tracks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("voice_name", help="e.g. anuv_jain (folder under reference_tracks/)")
    args = ap.parse_args()

    voice = args.voice_name
    tracks = find_tracks(voice)
    print(f"Found {len(tracks)} reference track(s) for '{voice}':")
    for t in tracks:
        print(f"  - {t.name}")

    work = config.DATASETS_DIR / voice / "_demucs_work"
    dataset = config.DATASETS_DIR / voice

    print("\n[1/2] Demucs: isolating vocals (GPU, a few minutes)...")
    vocals = isolate_vocals(tracks, work)

    print("[2/2] Slicing into training segments...")
    n = slice_segments(vocals, dataset)

    total_min = n * SEGMENT_SECONDS / 60
    print(f"\nDataset ready: {dataset}  ({n} segments, ~{total_min:.1f} min of vocals)")
    if total_min < 5:
        print("WARNING: less than 5 minutes of clean vocals - add more "
              "reference tracks for a decent clone (10-15 min is the sweet spot).")
    print(
        "\nNext: train the voice in Applio (see header of this script or the\n"
        "README 'Voice cloning' section), then copy the trained files to:\n"
        f"  {config.RVC_MODELS_DIR / (voice + '.pth')}\n"
        f"  {config.RVC_MODELS_DIR / (voice + '.index')}\n"
        f"\nThen use it:  python main.py --voice {voice}"
    )


if __name__ == "__main__":
    main()
