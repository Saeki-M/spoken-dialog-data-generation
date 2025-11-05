import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torchaudio


def load_audio_mono_16k(
    path: Path, target_sr: int = 16_000
) -> tuple[torch.Tensor, int, float]:
    wav, sr = torchaudio.load(str(path))  # (C, T), float32 in [-1,1]
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)  # mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    duration = wav.shape[-1] / float(sr)
    return wav.cpu(), sr, duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, type=Path, help="Path to WAV file")
    ap.add_argument("--text", required=True, help="Reference text to align")
    ap.add_argument(
        "--lang", default="ja", help="Language code for align model (e.g., ja, en)"
    )
    ap.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument(
        "--chars", action="store_true", help="Output character-level timings"
    )
    ap.add_argument(
        "--out", type=Path, help="Write JSON to this path (default: stdout)"
    )
    args = ap.parse_args()

    wav, sr, dur = load_audio_mono_16k(args.wav, 16_000)  # (1, T), 16 kHz
    audio_np = wav.squeeze(0).numpy()

    import whisperx

    align_model, meta = whisperx.load_align_model(
        language_code=args.lang, device=args.device
    )

    segments = [{"start": 0.0, "end": max(0.02, dur), "text": args.text}]

    aligned = whisperx.align(
        segments,
        align_model,
        meta,
        audio_np,
        args.device,
        args.chars,
    )

    unit_key = "chars" if args.chars else "words"
    label_key = "char" if args.chars else "word"

    items: List[Dict[str, Any]] = []
    for seg in aligned.get("segments", []):
        for w in seg.get(unit_key, []):
            if w.get("start") is None or w.get("end") is None:
                continue
            item = {
                label_key: w.get(label_key, ""),
                "start": float(w["start"]),
                "end": float(w["end"]),
            }
            if "score" in w and w["score"] is not None:
                item["score"] = float(w["score"])
            items.append(item)

    items.sort(key=lambda x: x["start"])

    out = {
        "audio_path": str(args.wav),
        "language": args.lang,
        "sample_rate": sr,
        "duration": dur,
        "unit": "char" if args.chars else "word",
        "items": items,
    }

    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
