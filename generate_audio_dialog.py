import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydub import AudioSegment

from tts import Vits

# ---------------------- CONFIG ---------------------- #
INPUT_DIR = Path("./output/text/")  # directory containing .tsv files
OUTPUT_DIR = Path("./output/audio/")  # where .wav/.json will be written
TIMESTAMP_DIR = Path("./output/timestamps/")
GLOB_PATTERN = "*.tsv"
OVERWRITE = False

INTER_TURN_SILENCE_MS = 400
NORMALIZE_TARGET_DBFS = -16.0

# One canonical audio format for everything
TARGET_FRAME_RATE = 24000
TARGET_SAMPLE_WIDTH = 2  # 16-bit
TARGET_CHANNELS = 1  # mono per side

# Role -> voice dir under ./vits_model/<voice>
voices: Dict[str, str] = {
    "assistant": "youtube1",
    "user": "youtube2",
}

# Speaker label mapping for JSON
SPEAKER_LABEL: Dict[str, str] = {
    "assistant": "A",
    "user": "B",
}

# For alignment
ALIGN_LANGUAGE = "ja"  # e.g., "ja", "en"
ALIGN_OUTPUT_UNIT = "char"  # "word" or "char"  (your example uses characters)
JSON_WORD_KEY = "word"  # keep key name as "word" even for char output, to match example
ROUND_MS = 3  # JSON time precision (e.g., 3 => milliseconds)

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


# ---------------------- Alignment (WhisperX) ---------------------- #
def _seg_to_numpy_16k(seg: AudioSegment):
    """Convert a pydub AudioSegment to mono 16k float32 NumPy array in [-1, 1]."""
    import numpy as np

    # ensure mono first (should already be)
    seg = seg.set_channels(1)
    # resample to 16k for aligner
    seg16 = seg.set_frame_rate(16_000)
    # get raw samples as ints
    samples = seg16.get_array_of_samples()
    np_int = np.array(samples)  # dtype usually int16
    # scale to float32 in [-1,1]
    if seg16.sample_width == 2:
        np_float = (np_int.astype("float32") / 32768.0).clip(-1.0, 1.0)
    else:
        # generic fallback
        max_abs = float(1 << (8 * seg16.sample_width - 1))
        np_float = (np_int.astype("float32") / max_abs).clip(-1.0, 1.0)
    return np_float, 16_000


class Aligner:
    """Lazy-initialized WhisperX aligner."""

    _loaded = False
    _align_model = None
    _meta = None
    _device = "cuda:0"

    @classmethod
    def ensure_loaded(cls, language_code: str):
        if cls._loaded:
            return
        try:
            import torch
            import whisperx

            cls._device = "cuda:0" if torch.cuda.is_available() else "cpu"
            cls._align_model, cls._meta = whisperx.load_align_model(
                language_code=language_code, device=cls._device
            )
            cls._loaded = True
        except Exception as e:
            raise RuntimeError(
                f"WhisperX alignment unavailable: {e}\n"
                "Install: pip install whisperx torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu124  (or CPU wheels)"
            )

    @classmethod
    def align_segment(
        cls, seg: AudioSegment, text: str, unit: str = "word"
    ) -> List[Dict[str, Any]]:
        """
        Returns list of {label, start, end} with times relative to the start of seg.
        unit: "word" or "char"
        """
        import whisperx

        if not text.strip():
            return []

        cls.ensure_loaded(ALIGN_LANGUAGE)
        audio_np, _ = _seg_to_numpy_16k(seg)

        # Wrap as a single "segment" for whisperx
        segments = [
            {"start": 0.0, "end": max(0.02, len(audio_np) / 16_000.0), "text": text}
        ]

        aligned = whisperx.align(
            segments,
            cls._align_model,
            cls._meta,
            audio_np,
            cls._device,
            (unit == "char"),
        )

        out: List[Dict[str, Any]] = []

        for word_timestamp in aligned["word_segments"]:
            if "start" in word_timestamp:
                out.append(
                    {
                        "label": word_timestamp["word"],
                        "start": float(word_timestamp["start"]),
                        "end": float(word_timestamp["end"]),
                    }
                )
        return out


# ---------------------- Core Synth + JSON ---------------------- #
def synthesize_stereo_conversation_and_words(
    rows: List[dict], engines: Dict[str, Vits]
) -> Tuple[AudioSegment, List[Dict[str, Any]]]:
    # Sort by turn_index if present
    try:
        rows = sorted(rows, key=lambda r: int(r["turn_index"]))
    except Exception:
        pass

    gap_frames = ms_to_frames(INTER_TURN_SILENCE_MS)

    left_track = silent_frames(0)
    right_track = silent_frames(0)
    is_first = True

    # We advance a global timeline by appending to both tracks in lockstep.
    # Global offset (in frames) is simply current length of the tracks.
    items_world: List[Dict[str, Any]] = []

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

        # Insert inter-turn gap (both channels) except before the very first segment
        if not is_first:
            gap = silent_frames(gap_frames)
            left_track += gap
            right_track += gap
        is_first = False

        # Compute global start offset (in seconds) BEFORE adding this segment
        # Current length (frames) is same for left and right because we always add in lockstep
        current_frames = int(round(left_track.frame_count()))
        global_offset_sec = current_frames / float(TARGET_FRAME_RATE)

        # Alignment (relative to seg) -> convert to world time
        aligned_local = Aligner.align_segment(seg, text, unit=ALIGN_OUTPUT_UNIT)
        speaker = SPEAKER_LABEL.get(role, SPEAKER_LABEL.get("assistant", "A"))
        for w in aligned_local:
            items_world.append(
                {
                    "speaker": speaker,
                    JSON_WORD_KEY: w["label"],
                    "start": round(global_offset_sec + w["start"], ROUND_MS),
                    "end": round(global_offset_sec + w["end"], ROUND_MS),
                }
            )

        # Add audio to the appropriate channel
        seg_frames = int(round(seg.frame_count()))
        if role == "user":
            left_track += silent_frames(seg_frames)
            right_track += seg
        else:
            left_track += seg
            right_track += silent_frames(seg_frames)

    # Final safety equalization by exact frame count
    left_track, right_track = equalize_frames(left_track, right_track)

    # Sort word/char items by world start time
    items_world.sort(key=lambda x: (x["start"], x["end"]))

    # Combine into stereo
    stereo = AudioSegment.from_mono_audiosegments(left_track, right_track)
    return stereo, items_world


# ---------------------- Batch Driver ---------------------- #
def process_one_tsv(
    tsv_path: Path, engines: Dict[str, Vits]
) -> Tuple[Path, Path | None]:
    rows = read_tsv(tsv_path)
    if not rows:
        raise ValueError(f"No rows in TSV: {tsv_path}")

    stereo_mix, items_world = synthesize_stereo_conversation_and_words(rows, engines)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TIMESTAMP_DIR.mkdir(parents=True, exist_ok=True)
    aac_path = OUTPUT_DIR / f"{tsv_path.stem}.mp3"
    json_path = TIMESTAMP_DIR / f"{tsv_path.stem}.json"

    stereo_mix.export(aac_path, format="mp3")

    if items_world:
        json_text = json.dumps(items_world, ensure_ascii=False, indent=2)
        json_path.write_text(json_text, encoding="utf-8")
        return aac_path, json_path
    else:
        print("⚠️  No alignment JSON produced (alignment disabled or failed).")
        return aac_path, None


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    tsvs = sorted(INPUT_DIR.glob(GLOB_PATTERN))
    if not tsvs:
        raise FileNotFoundError(f"No TSVs matched {GLOB_PATTERN} under {INPUT_DIR}")

    print(f"Found {len(tsvs)} TSV(s) in {INPUT_DIR}")

    # Build engines once and reuse for all files
    engines = build_tts_engines(voices)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    errors = []
    for i, tsv in enumerate(tsvs, start=1):
        print(f"[{i}/{len(tsvs)}] Processing {tsv.name} ...")

        out_audio = OUTPUT_DIR / f"{tsv.stem}.mp3"
        out_json = TIMESTAMP_DIR / f"{tsv.stem}.json"
        if out_audio.exists() and out_json.exists() and not OVERWRITE:
            print(f"⏭️  Skipping (exists): {out_audio}")
            continue

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
