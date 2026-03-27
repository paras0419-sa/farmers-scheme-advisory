"""
Voice-to-Text: Transcribe Hindi voice audio using Whisper fine-tuned for Hindi.

Converts Telegram voice notes (.ogg) to text via:
1. ffmpeg: .ogg → .wav (16kHz mono)
2. Whisper-Hindi: .wav → Hindi text

Usage:
    from src.voice import VoiceTranscriber

    transcriber = VoiceTranscriber()
    text = transcriber.transcribe("voice_note.ogg")
    print(text)
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


MODEL_ID = "vasista22/whisper-hindi-small"


class VoiceTranscriber:
    """Transcribes Hindi voice audio using Whisper fine-tuned for Hindi."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._pipe = None  # Lazy-load to avoid slow startup

    def _ensure_ffmpeg(self):
        """Check that ffmpeg is available."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg is not installed. Install it with: brew install ffmpeg"
            )

    def _load_model(self):
        """Load IndicWhisper model (lazy, first call only)."""
        if self._pipe is not None:
            return

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.float32  # MPS/CPU don't benefit from float16

        processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            dtype=torch_dtype,
        ).to(device)

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=torch_dtype,
            device=device,
        )
        print(f"Whisper-Hindi loaded on {device}")

    def ogg_to_wav(self, ogg_path: str) -> str:
        """Convert .ogg audio to 16kHz mono .wav for Whisper.

        Args:
            ogg_path: Path to the .ogg file.

        Returns:
            Path to the converted .wav file (in temp directory).
        """
        self._ensure_ffmpeg()

        wav_path = os.path.join(
            tempfile.gettempdir(),
            Path(ogg_path).stem + ".wav",
        )

        subprocess.run(
            [
                "ffmpeg", "-i", ogg_path,
                "-ar", "16000",   # 16kHz sample rate
                "-ac", "1",       # mono channel
                "-y",             # overwrite if exists
                wav_path,
            ],
            capture_output=True,
            check=True,
        )
        return wav_path

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text.

        Supports .ogg (auto-converts to .wav) and .wav files directly.

        Args:
            audio_path: Path to audio file (.ogg or .wav).

        Returns:
            Transcribed text string.
        """
        self._load_model()

        # Convert .ogg to .wav if needed
        if audio_path.endswith(".ogg"):
            wav_path = self.ogg_to_wav(audio_path)
        elif audio_path.endswith((".wav", ".mp3", ".flac")):
            wav_path = audio_path
        else:
            raise ValueError(f"Unsupported audio format: {audio_path}")

        result = self._pipe(wav_path)
        return result["text"].strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/voice.py <audio_file.ogg|.wav>")
        print("\nNo audio file provided. Testing model loading only...")
        transcriber = VoiceTranscriber()
        transcriber._load_model()
        print("Model loaded successfully!")
        sys.exit(0)

    audio_path = sys.argv[1]
    transcriber = VoiceTranscriber()
    text = transcriber.transcribe(audio_path)
    print(f"\nAudio: {audio_path}")
    print(f"Transcription: {text}")
