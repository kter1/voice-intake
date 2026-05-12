from __future__ import annotations

from .interface import ASRResult


class DeepgramAdapter:
    """
    Transcribes audio using Deepgram's nova-3 model via the Deepgram SDK v6+.
    Accepts raw audio bytes (e.g. WAV, mulaw from Twilio, mp3).

    Deepgram SDK v6 API:
        client = AsyncDeepgramClient(api_key)
        response = await client.listen.v1.media.transcribe_file(
            request=<bytes>,
            model="nova-3",
            language="en",
            punctuate=True,
            smart_format=True,
        )
        # response.results.channels is a list of channel items
        # channel.alternatives[0].transcript / .confidence
    """

    def __init__(self, api_key: str, model: str = "nova-3") -> None:
        self._api_key = api_key
        self._model = model

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> ASRResult:
        from deepgram import AsyncDeepgramClient  # type: ignore[import]

        client = AsyncDeepgramClient(self._api_key)
        response = await client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model=self._model,
            language=language,
            punctuate=True,
            smart_format=True,
        )
        try:
            channels = response.results.channels
            # channels may be a list or a RootModel - handle both
            channel = channels[0] if hasattr(channels, "__getitem__") else next(iter(channels))
            alt = channel.alternatives[0]
            transcript = alt.transcript or ""
            confidence = float(alt.confidence or 0.0)
        except (AttributeError, IndexError, StopIteration, TypeError):
            transcript = ""
            confidence = 0.0

        return ASRResult(
            transcript=transcript,
            confidence=confidence,
            is_final=True,
            language=language,
        )
