"""ACE-Step 1.5 wrapper tuned for 8 GB VRAM.

Uses the v1.5 API (AceStepHandler + LLMHandler + generate_music).
Loads the pipeline lazily, generates one song, then frees VRAM so Demucs
and RVC have room afterwards.
"""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path

from . import config


def generate_song(plan, out_path: Path) -> Path:
    """Render `plan` (a SongPlan) to an audio file at `out_path`."""
    # Must be set before acestep modules read them.
    os.environ.setdefault(
        "ACESTEP_CHECKPOINTS_DIR", str(config.ACESTEP_CHECKPOINT_DIR)
    )
    # torchcodec isn't installed on Windows; force the ffmpeg backend.
    os.environ.setdefault("TORCHAUDIO_USE_BACKEND", "ffmpeg")

    try:
        import acestep
        from acestep.handler import AceStepHandler
        from acestep.inference import (
            GenerationConfig,
            GenerationParams,
            generate_music,
        )
        from acestep.llm_inference import LLMHandler
        from acestep.model_downloader import ensure_lm_model
    except ImportError as e:
        raise RuntimeError(
            "ACE-Step is not installed in this venv. From the repo root run:\n"
            "  pip install --no-deps -e ..\\ACE-Step-1.5"
        ) from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    config.ACESTEP_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # The repo checkout is the project root (configs live there).
    repo_root = Path(acestep.__file__).resolve().parent.parent
    checkpoints = str(config.ACESTEP_CHECKPOINT_DIR)

    dit = AceStepHandler()
    llm = LLMHandler()
    try:
        print(f"      loading DiT model '{config.ACESTEP_CONFIG}' "
              "(downloads checkpoints on first run)...")
        status, ok = dit.initialize_service(
            project_root=str(repo_root),
            config_path=config.ACESTEP_CONFIG,
            device="cuda",
            offload_to_cpu=config.ACESTEP_OFFLOAD_CPU,
        )
        if not ok:
            raise RuntimeError(f"ACE-Step DiT init failed: {status}")

        print(f"      loading 5Hz LM '{config.ACESTEP_LM_MODEL}'...")
        dl_ok, dl_msg = ensure_lm_model(
            model_name=config.ACESTEP_LM_MODEL, checkpoints_dir=checkpoints
        )
        if not dl_ok:
            raise RuntimeError(f"ACE-Step LM download failed: {dl_msg}")
        # 'pt' backend: nano-vllm doesn't install on Windows.
        status, ok = llm.initialize(
            checkpoint_dir=checkpoints,
            lm_model_path=config.ACESTEP_LM_MODEL,
            backend="pt",
            device="cuda",
            offload_to_cpu=config.ACESTEP_OFFLOAD_CPU,
        )
        if not ok:
            raise RuntimeError(f"ACE-Step LM init failed: {status}")

        params = GenerationParams(
            caption=plan.style_tags,
            lyrics=plan.lyrics,
            duration=float(plan.duration_sec),
        )
        gen_config = GenerationConfig(batch_size=1, audio_format="wav")

        print("      generating audio...")
        result = generate_music(
            dit, llm, params, gen_config, save_dir=str(out_path.parent)
        )
        if not result.success or not result.audios:
            raise RuntimeError(f"ACE-Step generation failed: {result.error}")

        generated = Path(result.audios[0]["path"])
        if generated != out_path:
            shutil.move(str(generated), str(out_path))
        return out_path
    finally:
        del dit, llm
        _free_vram()


def _free_vram() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
