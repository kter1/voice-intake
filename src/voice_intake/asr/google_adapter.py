from __future__ import annotations

import asyncio

from .interface import ASRResult


class GoogleSpeechAdapter:
    """
    Fallback ASR via Google Cloud Speech-to-Text (sync SDK wrapped in asyncio.to_thread).
    Requires GOOGLE_APPLICATION_CREDENTIALS in the environment.
    """

    def __init__(self, sample_rate_hertz: int = 8000, encoding: str = "MULAW") -> None:
        self._sample_rate = sample_rate_hertz
        self._encoding = encoding

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> ASRResult:
        return await asyncio.to_thread(self._transcribe_sync, audio_bytes, language)

    def _transcribe_sync(self, audio_bytes: bytes, language: str) -> ASRResult:
        from google.cloud import speech  # type: ignore[import]

        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=getattr(speech.RecognitionConfig.AudioEncoding, self._encoding),
            sample_rate_hertz=self._sample_rate,
            language_code=language,
        )
        response = client.recognize(config=config, audio=audio)
        if not response.results:
            return ASRResult(transcript="", confidence=0.0, is_final=True, language=language)
        alt = response.results[0].alternatives[0]
        return ASRResult(
            transcript=alt.transcript,
            confidence=float(alt.confidence),
            is_final=True,
            language=language,
        )
