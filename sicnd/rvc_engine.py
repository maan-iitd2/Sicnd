"""RVC voice conversion: swap the generated vocal timbre for a cloned voice.

Voice models live in models/rvc/<voice_name>.pth (with an optional matching
<voice_name>.index built during training). Train them with clone_voice.py +
Applio — see README.
"""

from __future__ import annotations

import gc
from pathlib import Path

from . import config


def list_voices() -> list[str]:
    if not config.RVC_MODELS_DIR.exists():
        return []
    return sorted(p.stem for p in config.RVC_MODELS_DIR.glob("*.pth"))


def convert_vocals(
    vocals_in: Path,
    vocals_out: Path,
    voice_name: str,
    pitch_shift: int = 0,
) -> Path:
    """Run `vocals_in` through the RVC model `voice_name`."""
    pth = config.RVC_MODELS_DIR / f"{voice_name}.pth"
    if not pth.exists():
        available = ", ".join(list_voices()) or "(none - run clone_voice.py)"
        raise FileNotFoundError(
            f"No RVC model at {pth}. Available voices: {available}"
        )

    try:
        from rvc_python.infer import RVCInference
    except ImportError as e:
        raise RuntimeError(
            "rvc-python is not installed. Run: pip install rvc-python"
        ) from e

    index = config.RVC_MODELS_DIR / f"{voice_name}.index"

    rvc = RVCInference(device=config.RVC_DEVICE)
    try:
        rvc.load_model(
            str(pth),
            index_path=str(index) if index.exists() else "",
        )
        rvc.set_params(
            f0method=config.RVC_F0_METHOD,
            f0up_key=pitch_shift,
            index_rate=config.RVC_INDEX_RATE,
            protect=config.RVC_PROTECT,
        )
        vocals_out.parent.mkdir(parents=True, exist_ok=True)
        rvc.infer_file(str(vocals_in), str(vocals_out))
    finally:
        del rvc
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    if not vocals_out.exists():
        raise RuntimeError(f"RVC did not write {vocals_out}")
    return vocals_out
