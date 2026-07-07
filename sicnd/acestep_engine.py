"""ACE-Step wrapper tuned for 8 GB VRAM.

Loads the pipeline lazily, generates one song, then frees VRAM so Demucs
and RVC have room afterwards.
"""

from __future__ import annotations

import gc
from pathlib import Path

from . import config


def generate_song(plan, out_path: Path) -> Path:
    """Render `plan` (a SongPlan) to a wav file at `out_path`."""
    try:
        from ace_step.pipeline_ace_step import ACEStepPipeline
    except ImportError as e:
        raise RuntimeError(
            "ACE-Step is not installed. Run:\n"
            "  pip install git+https://github.com/ace-step/ACE-Step-1.5.git\n"
            "(see README section 'Install ACE-Step')."
        ) from e

    out_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = ACEStepPipeline(
        checkpoint_dir=str(config.ACESTEP_CHECKPOINT_DIR),
        dtype=config.ACESTEP_DTYPE,
        torch_compile=config.ACESTEP_TORCH_COMPILE,
        cpu_offload=config.ACESTEP_CPU_OFFLOAD,
        overlapped_decode=config.ACESTEP_OVERLAPPED_DECODE,
    )
    try:
        pipeline(
            prompt=plan.style_tags,
            lyrics=plan.lyrics,
            audio_duration=float(plan.duration_sec),
            infer_step=config.ACESTEP_INFER_STEPS,
            guidance_scale=config.ACESTEP_GUIDANCE_SCALE,
            save_path=str(out_path),
        )
    finally:
        del pipeline
        _free_vram()

    if not out_path.exists():
        # Some ACE-Step versions append their own suffix; pick up whatever
        # landed next to the requested path.
        candidates = sorted(
            out_path.parent.glob(out_path.stem + "*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                f"ACE-Step finished but no audio found at {out_path}"
            )
        return candidates[0]
    return out_path


def _free_vram() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
