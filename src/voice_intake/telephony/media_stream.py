"""WebSocket handler for Twilio Media Streams.

Protocol overview (Twilio → us):
  1. ``connected``  - initial handshake
  2. ``start``      - stream metadata (callSid, streamSid, mulaw 8 kHz mono)
  3. ``media``      - base64-encoded mulaw audio chunk (20 ms frames)
  4. ``stop``       - call ended

Live path:  mulaw frames → Deepgram streaming WS → final transcripts → turn pipeline
Mock path:  buffers all audio, decodes as UTF-8 text on ``stop`` (for tests/CI)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)

# Callback type: (transcript: str, is_final: bool) → None
TranscriptCallback = Callable[[str, bool], Coroutine[Any, Any, None]]


class MediaStreamHandler:
    """Drive a single Twilio Media Stream WebSocket connection.

    Parameters
    ----------
    session_id:
        Voice-intake session ID (opened by the inbound webhook before the
        stream starts).
    on_transcript:
        Async callback invoked whenever a transcript segment is available.
        Receives ``(transcript, is_final)``.
    """

    def __init__(self, session_id: str, on_transcript: TranscriptCallback) -> None:
        self._session_id = session_id
        self._on_transcript = on_transcript
        self.stream_sid: str | None = None
        self.call_sid: str | None = None
        self._audio_buffer: list[bytes] = []
        self._dg_connection: Any = None  # Deepgram async WS connection

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def handle(
        self,
        websocket: Any,
        api_key: str | None = None,
        mock_asr: bool = False,
    ) -> None:
        """Process all messages from the Twilio Media Stream WebSocket.

        Parameters
        ----------
        websocket:
            A FastAPI ``WebSocket`` that is already *accepted*.
        api_key:
            Deepgram API key.  Ignored in mock mode.
        mock_asr:
            When True, use the mock path (no Deepgram calls).
        """
        try:
            if mock_asr or not api_key:
                await self._handle_mock(websocket)
            else:
                await self._handle_live(websocket, api_key)
        except Exception:
            logger.exception(
                "MediaStreamHandler unhandled error session=%s", self._session_id
            )
        finally:
            if self._dg_connection is not None:
                try:
                    await self._dg_connection.finish()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Mock path - no Deepgram calls, suitable for tests / CI
    # ------------------------------------------------------------------

    async def _handle_mock(self, websocket: Any) -> None:
        """Accumulate audio chunks; on ``stop``, decode as UTF-8 and call back."""
        from fastapi.websockets import WebSocketDisconnect  # type: ignore[import]

        try:
            while True:
                raw = await websocket.receive_text()
                msg: dict[str, Any] = json.loads(raw)
                event = msg.get("event", "")

                if event == "connected":
                    logger.debug(
                        "Mock stream connected session=%s", self._session_id
                    )

                elif event == "start":
                    start = msg.get("start", msg)
                    self.stream_sid = start.get("streamSid") or msg.get("streamSid")
                    self.call_sid = start.get("callSid")
                    logger.info(
                        "Mock stream started session=%s callSid=%s",
                        self._session_id,
                        self.call_sid,
                    )

                elif event == "media":
                    payload = msg.get("media", {}).get("payload", "")
                    if payload:
                        self._audio_buffer.append(base64.b64decode(payload))

                elif event == "stop":
                    audio = b"".join(self._audio_buffer)
                    self._audio_buffer.clear()
                    transcript = audio.decode("utf-8", errors="replace").strip()
                    if transcript:
                        await self._on_transcript(transcript, True)
                    break

        except WebSocketDisconnect:
            logger.info("Mock stream disconnected session=%s", self._session_id)
        except Exception as exc:
            logger.exception(
                "Mock stream error session=%s: %s", self._session_id, exc
            )

    # ------------------------------------------------------------------
    # Live path - real Deepgram streaming
    # ------------------------------------------------------------------

    async def _handle_live(self, websocket: Any, api_key: str) -> None:
        """Stream mulaw audio to Deepgram; relay final transcripts to the pipeline."""
        from fastapi.websockets import WebSocketDisconnect  # type: ignore[import]

        dg_conn = await self._open_deepgram_stream(api_key)
        self._dg_connection = dg_conn

        try:
            while True:
                raw = await websocket.receive_text()
                msg: dict[str, Any] = json.loads(raw)
                event = msg.get("event", "")

                if event == "start":
                    start = msg.get("start", msg)
                    self.stream_sid = start.get("streamSid") or msg.get("streamSid")
                    self.call_sid = start.get("callSid")
                    logger.info(
                        "Live stream started session=%s callSid=%s",
                        self._session_id,
                        self.call_sid,
                    )

                elif event == "media":
                    payload = msg.get("media", {}).get("payload", "")
                    if payload:
                        audio_bytes = base64.b64decode(payload)
                        await dg_conn.send(audio_bytes)

                elif event == "stop":
                    await dg_conn.finish()
                    break

        except WebSocketDisconnect:
            logger.info("Live stream disconnected session=%s", self._session_id)
        except Exception as exc:
            logger.exception(
                "Live stream error session=%s: %s", self._session_id, exc
            )

    async def _open_deepgram_stream(self, api_key: str) -> Any:
        """Open a Deepgram live-transcription connection wired to our callback."""
        from deepgram import (  # type: ignore[import]
            AsyncDeepgramClient,
            LiveOptions,
            LiveTranscriptionEvents,
        )

        client = AsyncDeepgramClient(api_key)
        options = LiveOptions(
            model="nova-3",
            language="en-US",
            encoding="mulaw",
            sample_rate=8000,
            channels=1,
            punctuate=True,
            smart_format=True,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
        )

        connection = client.listen.asyncwebsocket.v("1")

        # Build a closure that captures ``self``
        on_transcript = self._on_transcript

        async def _on_message(self_conn: Any, result: Any, **kwargs: Any) -> None:  # noqa: ARG001
            try:
                alt = result.channel.alternatives[0]
                transcript: str = alt.transcript or ""
                is_final: bool = bool(result.is_final)
                if transcript:
                    await on_transcript(transcript, is_final)
            except Exception:
                logger.exception("Deepgram transcript handler error")

        connection.on(LiveTranscriptionEvents.Transcript, _on_message)

        started = await connection.start(options)
        if not started:
            raise RuntimeError("Deepgram streaming connection failed to start")

        return connection
