#!/usr/bin/env python
"""Sicnd - conversational local AI music studio.

Usage:
  python main.py                                    # interactive chat
  python main.py --voice anuv_jain                  # chat, clone this voice
  python main.py -p "sad indie song, acoustic"      # one-shot generation
  python main.py -p "..." --voice anuv_jain --pitch 0

Chat commands:
  (describe a song)   -> the LLM drafts lyrics + style tags for approval
  (more feedback)     -> refines the current draft
  go                  -> generate the approved draft
  voice <name>|off    -> pick/disable the RVC voice (see 'voices')
  voices              -> list trained RVC voices
  show                -> reprint the current draft
  quit                -> exit
"""

from __future__ import annotations

import argparse
import sys

from sicnd import config, pipeline, rvc_engine
from sicnd.llm import SongPlan, Songwriter

BANNER = r"""
  ___  _  ___  _  _  ___
 / __|| |/ __|| \| ||   \   local AI music studio
 \__ \| || (__| .` || |) |  describe a song, then type 'go'
 |___/|_|\___||_|\_||___/   'help' for commands
"""


def _check_ollama(writer: Songwriter) -> bool:
    try:
        writer.client.list()
        return True
    except Exception:
        print(
            "Cannot reach Ollama at "
            f"{config.OLLAMA_HOST}.\nStart it first: open the Ollama app or "
            "run `ollama serve`, and make sure you've pulled a model:\n"
            f"  ollama pull {config.OLLAMA_MODEL}"
        )
        return False


def _generate(plan: SongPlan, voice: str | None, pitch: int) -> None:
    if voice and voice not in rvc_engine.list_voices():
        print(
            f"Voice '{voice}' not found in {config.RVC_MODELS_DIR}. "
            f"Trained voices: {rvc_engine.list_voices() or 'none'}.\n"
            "Generating with ACE-Step's own vocals instead."
        )
        voice = None
    try:
        pipeline.run(plan, voice=voice, pitch_shift=pitch)
    except Exception as e:
        print(f"Generation failed: {e}")


def chat(voice: str | None, pitch: int) -> None:
    config.ensure_dirs()
    writer = Songwriter()
    if not _check_ollama(writer):
        sys.exit(1)

    print(BANNER)
    if voice:
        print(f"Voice cloning enabled: {voice}")
    plan: SongPlan | None = None

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        cmd, _, arg = user.partition(" ")
        cmd = cmd.lower()

        if cmd in ("quit", "exit", "q"):
            break
        if cmd == "help":
            print(__doc__)
        elif cmd == "voices":
            names = rvc_engine.list_voices()
            print("Trained voices: " + (", ".join(names) if names else
                  "none yet - run clone_voice.py (see README)"))
        elif cmd == "voice":
            if arg.lower() in ("off", "none", ""):
                voice = None
                print("Voice cloning disabled; ACE-Step vocals will be kept.")
            else:
                voice = arg.strip()
                print(f"Voice set to '{voice}'.")
        elif cmd == "show":
            print(plan.summary() if plan else "No draft yet - describe a song.")
        elif cmd in ("go", "generate", "make"):
            if not plan:
                print("Nothing to generate yet - describe a song first.")
                continue
            writer.unload()  # free LLM VRAM before ACE-Step loads
            _generate(plan, voice, pitch)
        else:
            # Anything else is songwriting input: draft or refine.
            print("thinking...")
            try:
                plan = writer.write(user)
            except Exception as e:
                print(f"LLM error: {e}")
                continue
            print("\n--- draft " + "-" * 50)
            print(plan.summary())
            print("-" * 60)
            print("Refine it with more feedback, or type 'go' to generate.")


def one_shot(prompt: str, voice: str | None, pitch: int) -> None:
    config.ensure_dirs()
    writer = Songwriter()
    if not _check_ollama(writer):
        sys.exit(1)
    print("Drafting song plan...")
    plan = writer.write(prompt)
    print(plan.summary())
    writer.unload()
    _generate(plan, voice, pitch)


def main() -> None:
    ap = argparse.ArgumentParser(description="Local conversational AI music studio")
    ap.add_argument("-p", "--prompt", help="one-shot: generate from this description and exit")
    ap.add_argument("--voice", help="RVC voice name from models/rvc/ (e.g. anuv_jain)")
    ap.add_argument("--pitch", type=int, default=0,
                    help="RVC pitch shift in semitones (e.g. 12 female->male source)")
    args = ap.parse_args()

    if args.prompt:
        one_shot(args.prompt, args.voice, args.pitch)
    else:
        chat(args.voice, args.pitch)


if __name__ == "__main__":
    main()
