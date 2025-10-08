#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TSV -> Simulated Conversation (Style-BERT-VITS2), Stereo Mix (frame-precise)

- assistant -> LEFT channel
- user      -> RIGHT channel
- Unknown roles fall back to assistant/LEFT
- Single output: <OUTPUT_DIR>/<TSV_STEM>.wav
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

from pydub import AudioSegment

from tts import Vits  # <-- change to your actual import path

# ---------------------- CONFIG ---------------------- #
TSV_PATH = Path("outputs_tsv/dialog_0007_QA.tsv")  # <-- point to your TSV
OUTPUT_DIR = Path("./tts_out")
INTER_TURN_SILENCE_MS = 400
NORMALIZE_TARGET_DBFS = -16.0

CONVERSATION_WAV = OUTPUT_DIR / f"{TSV_PATH.stem}.wav"

# One canonical audio format for everything
TARGET_FRAME_RATE = 24000
TARGET_SAMPLE_WIDTH = 2  # 16-bit
TARGET_CHANNELS = 1  # mono per side

voices: Dict[str, str] = {
    "assistant": "azusa",
    "user": "youtube2",
}

REMOVE_PARENS = False
# ---------------------------------------------------- #


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
    # Construct raw PCM silence
    return AudioSegment(
        data=b"\x00" * num_bytes,
        sample_width=TARGET_SAMPLE_WIDTH,
        frame_rate=TARGET_FRAME_RATE,
        channels=TARGET_CHANNELS,
    )


def ms_to_frames(ms: int) -> int:
    return int(round((ms / 1000.0) * TARGET_FRAME_RATE))


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

        # Inter-turn gap (exact frames) added to BOTH channels equally
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
    stereo = AudioSegment.from_mono_audiosegments(left_track, right_track)
    return stereo


def main():
    if not TSV_PATH.exists():
        raise FileNotFoundError(f"TSV not found: {TSV_PATH}")

    rows = read_tsv(TSV_PATH)
    if not rows:
        raise ValueError("No rows found in TSV.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engines = build_tts_engines(voices)
    stereo_mix = synthesize_stereo_conversation(rows, engines)

    stereo_mix.export(CONVERSATION_WAV, format="wav")
    print(f"✅ Done.\n- Conversation: {CONVERSATION_WAV}")


if __name__ == "__main__":
    main()
