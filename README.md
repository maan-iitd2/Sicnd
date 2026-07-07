# Sicnd — local conversational AI music studio

Talk to it, get a song back. Everything runs **fully locally** on your laptop —
no API keys, no cloud.

```
you> make me a sad indie song like Anuv Jain with acoustic guitar

--- draft --------------------------------------------------
Title      : Aadhi Raat
Style tags : indie folk, acoustic guitar, melancholic, soft hindi male
             vocals, intimate, fingerpicking, minimal production
Lyrics     : [verse] ...
------------------------------------------------------------
you> go
[1/4] ACE-Step: generating 'Aadhi Raat' (150s)...
[2/4] Demucs: splitting vocals from instrumental...
[3/4] RVC: converting vocals to voice 'anuv_jain'...
[4/4] ffmpeg: normalizing and exporting MP3...
Done -> outputs/20260707_193000_aadhi_raat/aadhi_raat.mp3
```

## How it works

| Stage | Tool | What it does |
|---|---|---|
| 1. Songwriting | **Ollama** (Llama 3.1 8B) | Your description → structured lyrics + style tags |
| 2. Music generation | **ACE-Step 1.5** | Tags + lyrics → full song (vocals + instrumental) |
| 3. Stem split | **Demucs** | Separates the song into vocals / instrumental |
| 4. Voice cloning | **RVC** | Replaces the vocal timbre with your cloned voice |
| 5. Narration (optional) | **Kokoro** | Spoken-word intro/outro |
| 6. Mix & export | **ffmpeg** | Remix, loudness-normalize, MP3 |

Stages load one at a time and free VRAM between steps, so the pipeline fits an
**RTX 4050 (8 GB VRAM)**.

## Repo layout

```
Sicnd/
├── main.py              # entry point: chat loop / one-shot CLI
├── clone_voice.py       # build an RVC training dataset from reference songs
├── sicnd/               # pipeline modules (llm, acestep, stems, rvc, mixer)
├── models/              # downloaded weights (gitignored)
│   └── rvc/             #   your trained voices: <name>.pth + <name>.index
├── outputs/             # generated songs (gitignored)
├── reference_tracks/    # artist songs for voice cloning (audio gitignored)
└── requirements.txt
```

## Setup (Windows + NVIDIA CUDA)

### 0. Prerequisites

```powershell
# Python 3.10 or 3.11 (3.12+ breaks some audio deps)
winget install Python.Python.3.11

# git + ffmpeg
winget install Git.Git
winget install Gyan.FFmpeg

# Ollama (local LLM server)
winget install Ollama.Ollama
```

Close and reopen your terminal after installing so PATH updates apply.
Make sure you have a recent NVIDIA driver (CUDA 12.x capable — anything from
2024+ is fine; check with `nvidia-smi`).

### 1. Clone and create a virtual env

```powershell
git clone https://github.com/maan-iitd2/Sicnd
cd Sicnd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
```

(Prefer [uv](https://docs.astral.sh/uv/)? `uv venv --python 3.11` then
`uv pip install ...` works for every pip command below.)

### 2. Install PyTorch with CUDA (must come first)

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

### 3. Install everything else

```powershell
pip install -r requirements.txt
```

This pulls ACE-Step 1.5 from GitHub, plus Demucs, rvc-python, Kokoro, and the
Ollama client. If the ACE-Step git install fails, install it manually:

```powershell
git clone https://github.com/ace-step/ACE-Step-1.5 ..\ACE-Step-1.5
pip install -e ..\ACE-Step-1.5
```

### 4. Use an existing Ollama model

The default is `qwen2.5:7b-instruct` (4.7 GB, great for songwriting instructions).
Use a different model by setting the env var:

```powershell
# See what you have
ollama list

# Use a specific model
set SICND_OLLAMA_MODEL=qwen3:8b
python main.py

# Or pull a new one first
ollama pull mistral
set SICND_OLLAMA_MODEL=mistral
```

If you don't have any models yet:
```powershell
ollama pull qwen2.5:7b-instruct
```

### 5. First run

```powershell
python main.py
```

- ACE-Step downloads its checkpoints (~7 GB) into `models/acestep/` on first
  generation — one-time, be patient.
- On 8 GB VRAM the first song takes several minutes; later runs are faster.

## Usage

**Chat mode** (default):

```powershell
python main.py                       # ACE-Step's own vocals
python main.py --voice anuv_jain     # with a cloned voice
```

Describe a song, refine the draft with follow-up messages ("make the chorus
in Hindi", "slower, more nostalgic"), then type `go`. Other commands:
`voices`, `voice <name>`, `voice off`, `show`, `quit`.

**One-shot mode**:

```powershell
python main.py -p "dreamy lo-fi bedroom pop about monsoon, female vocals" --voice anuv_jain
```

Output lands in `outputs/<timestamp>_<title>/` with the final `.mp3`, the
lyrics, and all intermediate stems in `work/` (handy for remixing by hand).

## Voice cloning (e.g. Anuv Jain style)

1. **Collect references**: put 3–6 songs (10–15 min of clean vocals) in
   `reference_tracks/anuv_jain/`.
2. **Build the dataset** (automated — Demucs isolates the vocals, then they're
   sliced into clean training segments):

   ```powershell
   python clone_voice.py anuv_jain
   ```

3. **Train the RVC model** — one-time, ~30–60 min on the 4050. Training needs
   an interactive trainer; [Applio](https://github.com/IAHispano/Applio) is
   the easiest fully-local option:
   - download an Applio release, run `run-install.bat`, then `run-applio.bat`
   - Train tab → dataset path: `datasets\anuv_jain` → sample rate **40k** →
     ~250 epochs
   - copy the exported files into this repo:
     `models/rvc/anuv_jain.pth` and `models/rvc/anuv_jain.index`
4. **Use it**: `python main.py --voice anuv_jain`. If the generated vocals sit
   in the wrong octave for the cloned voice, nudge with `--pitch -12` / `--pitch 12`.

> Voice cloning is for private, personal experiments — only clone voices you
> have the right to use.

## 8 GB VRAM notes

Already configured in `sicnd/config.py`:

- ACE-Step runs with `bfloat16` + `cpu_offload` + `overlapped_decode`
- the Ollama model is unloaded from VRAM before ACE-Step starts
- Demucs and RVC run sequentially after ACE-Step releases the GPU
- default song length 150 s — keep `duration_sec ≤ 180` on 8 GB

If you still hit CUDA OOM: close anything else using the GPU (browsers with
hardware acceleration count), shorten the song ("make it 90 seconds"), or
lower `ACESTEP_INFER_STEPS` in `sicnd/config.py`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot reach Ollama` | Start the Ollama app (or `ollama serve`), then `ollama pull llama3.1:8b` |
| `ffmpeg not found` | `winget install Gyan.FFmpeg`, reopen terminal |
| `torch.cuda.is_available()` is `False` | Reinstall torch with the `--index-url .../cu121` flag; update NVIDIA driver |
| ACE-Step import error | Install manually (step 3 fallback); check the [ACE-Step 1.5 repo](https://github.com/ace-step/ACE-Step-1.5) README for API changes |
| Kokoro voice errors | `pip install kokoro soundfile`; on some setups also `pip install espeakng-loader` |
| RVC output sounds robotic | Add more/cleaner reference vocals, retrain with more epochs, try `--pitch` adjustments |
