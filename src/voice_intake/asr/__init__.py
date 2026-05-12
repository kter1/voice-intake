from .deepgram_adapter import DeepgramAdapter
from .google_adapter import GoogleSpeechAdapter
from .interface import ASRInterface, ASRResult

__all__ = ["ASRInterface", "ASRResult", "DeepgramAdapter", "GoogleSpeechAdapter"]


def get_asr_adapter(settings: object) -> "ASRInterface":
    """
    Factory: returns the appropriate ASR adapter based on settings.
    Falls back to mock_asr if mock_asr=True (returns the MockASR shim).
    """
    mock_asr = getattr(settings, "mock_asr", False)
    if mock_asr:
        from voice_intake.testing.mock_asr import MockASRAdapter
        return MockASRAdapter()

    deepgram_key = getattr(settings, "deepgram_api_key", "")
    if deepgram_key:
        return DeepgramAdapter(api_key=deepgram_key)

    raise RuntimeError(
        "No ASR adapter configured. Set DEEPGRAM_API_KEY or MOCK_ASR=true."
    )
