from __future__ import annotations

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .security import verify_api_key

_TWILIO_WEBHOOK = "/telephony/twilio/voice"


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self._key = api_key

    async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
        if not self._key:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path == _TWILIO_WEBHOOK:
            return await call_next(request)
        provided = request.headers.get("X-API-Key")
        if not verify_api_key(provided, self._key):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)
