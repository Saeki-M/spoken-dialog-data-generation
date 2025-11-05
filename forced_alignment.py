#!/usr/bin/env python3
# align_jp.py
"""
Minimal forced-alignment CLI for WAV + transcript -> JSON word timings.

Requirements (install one-time):
    pip install whisperx torch

Example:
    python align_jp.py --wav /path/audio.wav --text "今日はいい天気ですね"
    python align_jp.py --wav /path/audio.wav --text-file /path/text.txt --out out.json

Notes:
- Defaults to Japanese ('ja'). Works on CPU or CUDA automatically.
- Output JSON (stdout or --out) is a list of items:
    [{"word": "...", "start": 0.12, "end": 0.30, "score": 0.91}]
"""

import argparse
import json
import os
import sys
import unicodedata
import wave
from contextlib import closing

import torch
import whisperx  # type: ignore


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def wav_duration_seconds(wav_path: str) -> float:
    """Read WAV duration using stdlib 'wave' (PCM WAV recommended)."""
    with closing(wave.open(wav_path, "rb")) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate == 0:
            return 0.0
        return round(frames / float(rate), 6)


def normalize_text_jp(s: str) -> str:
    """
    Light normalization similar to user's pipeline:
    - lower-case (mostly impacts latin)
    - convert full-width ASCII to half-width
    - strip surrounding whitespace
    (Avoid aggressive token/space changes for Japanese.)
    """
    s = s.strip().lower()
    # Convert full-width ASCII/punctuation to half-width
    s = unicodedata.normalize("NFKC", s)
    return s


def load_align_model(language_code: str, device: str, model_name: str | None):
    """
    WhisperX align model loader with a safe fallback if model_name is unsupported.
    """
    if model_name:
        try:
            return whisperx.load_align_model(
                language_code=language_code, device=device, model_name=model_name
            )
        except TypeError:
            # Older whisperx versions may not accept model_name kwarg
            pass
        except Exception:
            # If the requested model_name fails, fall back
            pass
    # Fallback to default model for the language
    return whisperx.load_align_model(language_code=language_code, device=device)


def run_align(wav_path: str, text: str, lang: str, device: str, model_name: str | None):
    # Prepare single segment spanning the full file
    dur = wav_duration_seconds(wav_path)
    if dur <= 0:
        raise ValueError(
            "Could not read a valid duration from WAV. Ensure it's a standard PCM WAV file."
        )
    segs = [
        {
            "text": normalize_text_jp(text) if lang == "ja" else text.strip(),
            "start": 0.0,
            "end": float(dur),
        }
    ]

    model_a, metadata = load_align_model(lang, device, model_name)

    # Perform alignment
    # whisperx.align returns a dict with "word_segments" among other things
    aligned = whisperx.align(segs, model_a, metadata, wav_path, device)

    # Build a compact list of word timings
    out = []
    for w in aligned.get("word_segments", []):
        # Some versions may include additional keys; we just keep the essentials
        word = w.get("word")
        start = w.get("start")
        end = w.get("end")
        score = w.get("score")
        if word is None or start is None or end is None:
            continue
        out.append(
            {
                "word": word,
                "start": float(start),
                "end": float(end),
                "score": float(score) if score is not None else None,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Forced alignment (WAV + text) -> JSON word timings."
    )
    ap.add_argument(
        "--wav", required=True, help="Path to input WAV file (PCM WAV recommended)."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Transcript text (string).")
    g.add_argument("--text-file", help="Path to a UTF-8 text file with transcript.")
    ap.add_argument(
        "--lang", default="ja", help="Language code for the align model (default: ja)."
    )
    ap.add_argument(
        "--model-name",
        default=None,
        help="Optional specific align model name (e.g., reazon-research/japanese-wav2vec2-large-rs35kh).",
    )
    ap.add_argument("--out", default=None, help="Output JSON path (default: stdout).")
    args = ap.parse_args()

    if not os.path.isfile(args.wav):
        ap.error(f"--wav not found: {args.wav}")

    if args.text_file:
        if not os.path.isfile(args.text_file):
            ap.error(f"--text-file not found: {args.text_file}")
        with open(args.text_file, "r", encoding="utf-8") as f:
            transcript = f.read().strip()
    else:
        transcript = (args.text or "").strip()

    if not transcript:
        ap.error("Transcript is empty. Provide --text or --text-file with content.")

    device = get_device()
    try:
        result = run_align(args.wav, transcript, args.lang, device, args.model_name)
    except Exception as e:
        print(f"Alignment failed: {e}", file=sys.stderr)
        sys.exit(2)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(
            args.out
        ) else None
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=2)
    else:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        print()


if __name__ == "__main__":
    main()
