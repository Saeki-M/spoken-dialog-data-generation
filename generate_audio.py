#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Folder TSVs -> Stereo Conversation WAVs (frame-precise, robust)

- assistant -> LEFT channel
- user      -> RIGHT channel
- Unknown roles fall back to assistant/LEFT
- For each TSV in INPUT_DIR, output <OUTPUT_DIR>/<tsv_stem>.wav
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

from pydub import AudioSegment

from tts import Vits

# ---------------------- CONFIG ---------------------- #
INPUT_DIR = Path("./output/transcript/")  # directory containing .tsv files
OUTPUT_DIR = Path("./output/audio/")  # where .wav files will be written
GLOB_PATTERN = "*.tsv"  # which TSVs to process
OVERWRITE = True  # set False to skip if output exists

INTER_TURN_SILENCE_MS = 400
NORMALIZE_TARGET_DBFS = -16.0

# One canonical audio format for everything
TARGET_FRAME_RATE = 24000
TARGET_SAMPLE_WIDTH = 2  # 16-bit
TARGET_CHANNELS = 1  # mono per side

# Role -> voice dir under ./vits_model/<voice>
voices: Dict[str, str] = {
    "assistant": "azusa",
    "user": "youtube2",
}

# Optional: strip content in parentheses?
REMOVE_PARENS = False
# ---------------------------------------------------- #


# ---------------------- I/O & Text ---------------------- #
def read_tsv(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows


def prepare_text(s: str) -> str:
    s = s.strip()
    if REMOVE_PARENS:
        s = re.sub(r"（.*?）|\(.*?\)", "", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
    return s


# ---------------------- Audio Utils ---------------------- #
def normalize(segment: AudioSegment, target_dbfs: float) -> AudioSegment:
    change = target_dbfs - segment.dBFS if segment.dBFS != float("-inf") else 0.0
    return segment.apply_gain(change)


def to_target_format(seg: AudioSegment) -> AudioSegment:
    """Ensure exact target format (rate/width/channels=mono)."""
    return (
        seg.set_frame_rate(TARGET_FRAME_RATE)
        .set_sample_width(TARGET_SAMPLE_WIDTH)
        .set_channels(TARGET_CHANNELS)
    )


def silent_frames(n_frames: int) -> AudioSegment:
    """Create exact-length silence by frame count (no ms rounding)."""
    if n_frames <= 0:
        return (
            AudioSegment.silent(duration=0, frame_rate=TARGET_FRAME_RATE)
            .set_sample_width(TARGET_SAMPLE_WIDTH)
            .set_channels(1)
        )
    num_bytes = n_frames * TARGET_SAMPLE_WIDTH * TARGET_CHANNELS
    return AudioSegment(
        data=b"\x00" * num_bytes,
        sample_width=TARGET_SAMPLE_WIDTH,
        frame_rate=TARGET_FRAME_RATE,
        channels=TARGET_CHANNELS,
    )


def ms_to_frames(ms: int) -> int:
    return int(round((ms / 1000.0) * TARGET_FRAME_RATE))


def equalize_frames(
    left: AudioSegment, right: AudioSegment
) -> tuple[AudioSegment, AudioSegment]:
    """Pad the shorter side with exact frames of silence to match lengths."""
    lf = int(round(left.frame_count()))
    rf = int(round(right.frame_count()))
    if lf == rf:
        return left, right
    if lf < rf:
        left += silent_frames(rf - lf)
    else:
        right += silent_frames(lf - rf)
    return left, right


# ---------------------- TTS Engines ---------------------- #
def build_tts_engines(voices_map: Dict[str, str]) -> Dict[str, Vits]:
    engines: Dict[str, Vits] = {}
    missing: List[Tuple[str, str]] = []

    for role, voice in voices_map.items():
        try:
            engines[role] = Vits(language_code="ja-JP", name=voice)
        except Exception:
            missing.append((role, voice))
    if missing:
        lines = [f"- role '{r}': voice '{v}'" for r, v in missing]
        raise RuntimeError(
            "The following role->voice models could not be loaded:\n"
            + "\n".join(lines)
            + "\nMake sure ./vits_model/<voice>/ exists with .safetensors, config.json, style_vectors.npy."
        )
    return engines


# ---------------------- Core Synth ---------------------- #
def synthesize_stereo_conversation(
    rows: List[dict], engines: Dict[str, Vits]
) -> AudioSegment:
    # Sort by turn_index if present
    try:
        rows = sorted(rows, key=lambda r: int(r["turn_index"]))
    except Exception:
        pass

    gap_frames = ms_to_frames(INTER_TURN_SILENCE_MS)

    left_track = silent_frames(0)
    right_track = silent_frames(0)
    is_first = True

    for r in rows:
        role = (r.get("role") or "").strip()
        text = prepare_text(r.get("content") or "")
        if not text:
            continue

        engine_key = (
            role
            if role in engines
            else ("assistant" if "assistant" in engines else next(iter(engines.keys())))
        )
        seg = engines[engine_key].generate_audiosegment(
            text, speed=1.0, pitch=0.0, volume_gain=0.0
        )
        seg = to_target_format(normalize(seg, NORMALIZE_TARGET_DBFS))

        if not is_first:
            gap = silent_frames(gap_frames)
            left_track += gap
            right_track += gap
        is_first = False

        seg_frames = int(round(seg.frame_count()))

        if role == "user":
            left_track += silent_frames(seg_frames)
            right_track += seg
        else:
            left_track += seg
            right_track += silent_frames(seg_frames)

    # Final safety equalization by exact frame count
    left_track, right_track = equalize_frames(left_track, right_track)

    # Combine into stereo
    return AudioSegment.from_mono_audiosegments(left_track, right_track)


# ---------------------- Batch Driver ---------------------- #
def process_one_tsv(tsv_path: Path, engines: Dict[str, Vits]) -> Path:
    rows = read_tsv(tsv_path)
    if not rows:
        raise ValueError(f"No rows in TSV: {tsv_path}")

    stereo_mix = synthesize_stereo_conversation(rows, engines)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{tsv_path.stem}.wav"

    if out_path.exists() and not OVERWRITE:
        print(f"⏭️  Skipping (exists): {out_path}")
        return out_path

    stereo_mix.export(out_path, format="wav")
    print(f"✅ Wrote: {out_path}")
    return out_path


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    tsvs = sorted(INPUT_DIR.glob(GLOB_PATTERN))
    if not tsvs:
        raise FileNotFoundError(f"No TSVs matched {GLOB_PATTERN} under {INPUT_DIR}")

    print(f"Found {len(tsvs)} TSV(s) in {INPUT_DIR}")

    # Build engines once and reuse for all files
    engines = build_tts_engines(voices)

    errors = []
    for i, tsv in enumerate(tsvs, start=1):
        print(f"[{i}/{len(tsvs)}] Processing {tsv.name} ...")
        try:
            process_one_tsv(tsv, engines)
        except Exception as e:
            errors.append((tsv, e))
            print(f"❌ Error: {tsv} -> {e}")

    if errors:
        print("\nCompleted with errors on these files:")
        for p, e in errors:
            print(f"- {p}: {e}")
    else:
        print("\n🎉 All TSVs processed successfully.")


if __name__ == "__main__":
    main()
