/**
 * VoiceInputPanel - browser-based speech input with graceful text fallback.
 *
 * Chrome / Edge:  Uses the Web Speech API (SpeechRecognition).
 *                 Zero cost, zero API keys. Tap to speak, tap again to stop.
 * Other browsers: Falls back to a plain textarea + Send button.
 *
 * Props:
 *   onTranscript  called with the final recognised text
 *   disabled      disable input while a request is in-flight
 */

import { useEffect, useRef, useState } from "react";

interface Props {
  onTranscript: (text: string) => void;
  disabled?: boolean;
}

// Grab the SpeechRecognition constructor from the window - works on Chrome/Edge.
// We use `any` because Web Speech API types aren't consistently available across TS versions.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const SpeechRecognitionCtor: (new () => any) | undefined =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (window as any).SpeechRecognition ?? (window as any).webkitSpeechRecognition;

function hasSpeechRecognition(): boolean {
  return !!SpeechRecognitionCtor;
}

export function VoiceInputPanel({ onTranscript, disabled = false }: Props) {
  const [listening, setListening] = useState(false);
  const [interimText, setInterimText] = useState("");
  const [textInput, setTextInput] = useState("");
  const [speechSupported] = useState(hasSpeechRecognition);
  const [forceText, setForceText] = useState(false);
  const [recognitionError, setRecognitionError] = useState<string | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recognitionRef = useRef<any>(null);
  const accumulatedRef = useRef("");
  const shouldSubmitOnEndRef = useRef(false);
  const showText = !speechSupported || forceText;

  useEffect(() => {
    return () => {
      recognitionRef.current?.abort();
    };
  }, []);

  // ── Chrome / Edge path ────────────────────────────────────────────────────

  function startListening() {
    if (!SpeechRecognitionCtor) return;
    recognitionRef.current?.abort();
    setInterimText("");
    setRecognitionError(null);
    accumulatedRef.current = "";
    shouldSubmitOnEndRef.current = false;

    const recognition = new SpeechRecognitionCtor();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = true;
    recognitionRef.current = recognition;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onresult = (e: any) => {
      let interim = "";
      for (let i = e.resultIndex; i < (e.results as ArrayLike<unknown>).length; i++) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const result = (e.results as any)[i];
        if (result.isFinal) {
          const segment = (result[0].transcript as string).trim();
          accumulatedRef.current = accumulatedRef.current
            ? `${accumulatedRef.current} ${segment}`
            : segment;
        } else {
          interim = result[0].transcript as string;
        }
      }
      const composed = [accumulatedRef.current, interim].filter(Boolean).join(" ").trim();
      setInterimText(composed);
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onerror = (event: any) => {
      setListening(false);
      setInterimText("");
      accumulatedRef.current = "";
      shouldSubmitOnEndRef.current = false;
      const code: string = event?.error ?? "unknown";
      setRecognitionError(
        `Microphone unavailable (${code}). You can type instead.`
      );
      setForceText(true);
    };

    recognition.onend = () => {
      setListening(false);
      setInterimText("");
      if (shouldSubmitOnEndRef.current) {
        const text = accumulatedRef.current.trim();
        accumulatedRef.current = "";
        shouldSubmitOnEndRef.current = false;
        if (text) {
          onTranscript(text);
        }
        return;
      }
      accumulatedRef.current = "";
    };

    try {
      recognition.start();
      setListening(true);
    } catch {
      setRecognitionError("Microphone could not start. You can type instead.");
      setForceText(true);
    }
  }

  function stopListening() {
    shouldSubmitOnEndRef.current = true;
    recognitionRef.current?.stop();
    setListening(false);
    setInterimText("");
  }

  function handleMicClick() {
    if (disabled) return;
    if (listening) {
      stopListening();
      return;
    }
    startListening();
  }

  // ── Text fallback path ────────────────────────────────────────────────────

  function handleSend() {
    const t = textInput.trim();
    if (!t) return;
    setTextInput("");
    onTranscript(t);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────

  if (!showText) {
    return (
      <div className="flex flex-col items-center gap-3 py-4">
        {interimText && (
          <p className="text-sm text-gray-500 italic">"{interimText}…"</p>
        )}
        {recognitionError && (
          <p className="text-xs text-red-500">{recognitionError}</p>
        )}
        <button
          onClick={handleMicClick}
          aria-pressed={listening}
          disabled={disabled}
          className={[
            "w-20 h-20 rounded-full text-3xl shadow-lg transition-all select-none",
            listening
              ? "bg-red-500 text-white scale-110 ring-4 ring-red-300 animate-pulse"
              : "bg-blue-600 text-white hover:bg-blue-700 active:scale-95",
            disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer",
          ].join(" ")}
          title={listening ? "Stop microphone" : "Use microphone"}
        >
          {listening ? "🔴" : "🎙"}
        </button>
        <p className="text-xs text-gray-400">
          {listening ? "Tap to stop" : disabled ? "Waiting…" : "Tap to speak"}
        </p>
        <p className="text-xs text-gray-300">
          or{" "}
          <button
            className="underline"
            onClick={() => setForceText(true)}
            disabled={disabled}
          >
            type instead
          </button>
        </p>
      </div>
    );
  }

  // Text fallback
  return (
    <div className="flex flex-col gap-2">
      {recognitionError && (
        <p className="text-xs text-red-500 px-1">{recognitionError}</p>
      )}
      <div className="flex gap-2 items-end">
        <textarea
          rows={2}
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder="Type a message and press Enter (or click Send)…"
          className="flex-1 border rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:opacity-50"
        />
        {speechSupported && (
          <button
            onClick={() => setForceText(false)}
            disabled={disabled}
            className="px-4 py-2 border rounded-lg text-sm font-medium hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Use microphone
          </button>
        )}
        <button
          onClick={handleSend}
          disabled={disabled || !textInput.trim()}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {disabled ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}
