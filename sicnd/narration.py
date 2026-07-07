"""Kokoro TTS for spoken-word intros/outros."""

from __future__ import annotations

from pathlib import Path

from . import config


def synthesize(text: str, out_path: Path) -> Path:
    try:
        from kokoro import KPipeline
    except ImportError as e:
        raise RuntimeError(
            "Kokoro is not installed. Run: pip install kokoro soundfile"
        ) from e

    import numpy as np
    import soundfile as sf

    pipeline = KPipeline(lang_code=config.KOKORO_LANG)
    chunks = [
        audio for _, _, audio in pipeline(text, voice=config.KOKORO_VOICE)
    ]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), np.concatenate(chunks), config.KOKORO_SAMPLE_RATE)
    return out_path
