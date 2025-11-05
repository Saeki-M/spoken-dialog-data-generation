import io
from pathlib import Path

import style_bert_vits2
import style_bert_vits2.logging
import torch
from pydub import AudioSegment
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel

style_bert_vits2.logging.logger.remove()


class Vits:
    def __init__(
        self, language_code="ja-JP", sample_rate=32000, name: str = "azusa", pitch=0
    ):
        assert language_code in ["ja-JP"], "Only support ja-JP for now"

        bert_models.load_model(
            Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm"
        )
        bert_models.load_tokenizer(
            Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm"
        )

        asset_dir = Path(__file__).parent / "./vits_model" / name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TTSModel(
            model_path=asset_dir / f"{name}.safetensors",
            config_path=asset_dir / "config.json",
            style_vec_path=asset_dir / "style_vectors.npy",
            device=device,
        )
        self.model.load()
        self.style = "Neutral"

        self.generate_audiosegment(
            "こんにちは。", 1.0, 0.0, 0.0
        )  # first generation is slow, so warm up the model

    def clean_text(self, text):
        text = text.replace("\n", "")
        return text

    def generate_audiobytes(
        self,
        text: str,
        speed: float,
        pitch: float,
        volume_gain_db: float,
        lang: str | None = None,
    ) -> tuple[bytes, list]:
        """Currently StyleBertVITS2 does not support speed, pitch, and volume gain"""
        sr, audio = self.model.infer(text=text, style=self.style)
        raise NotImplementedError("StyleBertVITS2 does not support generate_audiobytes")

    def generate_audiosegment(self, text, speed, pitch, volume_gain, lang=None):
        """Currently StyleBertVITS2 does not support pitch and volume gain"""
        text = self.clean_text(text)
        sr, audio = self.model.infer(text=text, style=self.style)
        segment: AudioSegment = AudioSegment(
            audio.tobytes(), frame_rate=sr, sample_width=2, channels=1
        )

        wav_io = io.BytesIO()
        segment.export(wav_io, format="wav")

        return segment

    def generate_to_file(
        self, text, save_path, speed=1, pitch=1, volume_gain_db=0, lang=None
    ):
        save_path = str(save_path)
        audio_segment = self.generate_audiosegment(
            text, speed, pitch, volume_gain_db, lang
        )
        audio_segment.export(save_path, format=save_path.split(".")[-1])


if __name__ == "__main__":
    vits = Vits(name="youtube1")
    vits.generate_to_file("こんにちは、元気ですか？", "output.wav")
