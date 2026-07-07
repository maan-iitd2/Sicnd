"""Central configuration. Everything is a plain constant so the project
stays hackable; override via environment variables where noted."""

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REFERENCE_DIR = ROOT_DIR / "reference_tracks"
DATASETS_DIR = ROOT_DIR / "datasets"

ACESTEP_CHECKPOINT_DIR = MODELS_DIR / "acestep"   # ACE-Step downloads here
RVC_MODELS_DIR = MODELS_DIR / "rvc"               # <voice>.pth / <voice>.index

# ---------------------------------------------------------------- ollama
# Any model you've pulled works. qwen2.5:7b-instruct is the default —
# good at following instructions for songwriting. You can override:
#   set SICND_OLLAMA_MODEL=qwen3:8b  (or mistral, gemma3:4b, etc.)
OLLAMA_MODEL = os.environ.get("SICND_OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# ---------------------------------------------------------------- ace-step
# v1.5 API. CPU offload keeps the DiT + 5Hz LM within 8 GB VRAM (RTX 4050);
# the 0.6B LM is the smallest that still handles CoT metadata/caption work.
ACESTEP_CONFIG = os.environ.get("SICND_ACESTEP_CONFIG", "acestep-v15-turbo")
ACESTEP_LM_MODEL = os.environ.get("SICND_ACESTEP_LM", "acestep-5Hz-lm-0.6B")
ACESTEP_OFFLOAD_CPU = True
DEFAULT_DURATION_SEC = 150          # ~2.5 min; keep <=180 on 8 GB VRAM

# ---------------------------------------------------------------- demucs
DEMUCS_MODEL = "htdemucs"           # two-stem vocal split, fits 8 GB easily

# ---------------------------------------------------------------- rvc
RVC_F0_METHOD = "rmvpe"             # best quality pitch extraction
RVC_INDEX_RATE = 0.66
RVC_PROTECT = 0.33                  # protects breaths/consonants
RVC_DEVICE = os.environ.get("SICND_RVC_DEVICE", "cuda:0")

# ---------------------------------------------------------------- kokoro
KOKORO_VOICE = "af_heart"           # narration voice; see kokoro docs for more
KOKORO_LANG = "a"                   # 'a' = American English
KOKORO_SAMPLE_RATE = 24000

# ---------------------------------------------------------------- mixing
FINAL_SAMPLE_RATE = 44100
MP3_QUALITY = "2"                   # libmp3lame -q:a 2 (~190 kbps VBR)


def ensure_dirs() -> None:
    for d in (MODELS_DIR, OUTPUTS_DIR, REFERENCE_DIR, RVC_MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
