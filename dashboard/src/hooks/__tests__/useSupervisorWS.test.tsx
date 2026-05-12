import { renderHook, act } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  readonly url: string;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: ((event: { code: number }) => void) | null = null;
  close = vi.fn();

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  emitMessage(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }

  emitClose(code: number) {
    this.onclose?.({ code });
  }
}

describe("useSupervisorWS", () => {
  afterEach(() => {
    MockWebSocket.instances = [];
    vi.useRealTimers();
    vi.resetModules();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("includes api_key in the websocket URL when configured", async () => {
    vi.stubEnv("VITE_API_KEY", "pilot-secret");
    vi.stubGlobal("WebSocket", MockWebSocket);

    const { useSupervisorWS } = await import("../useSupervisorWS");
    renderHook(() => useSupervisorWS());

    expect(MockWebSocket.instances).toHaveLength(1);
    expect(MockWebSocket.instances[0]?.url).toContain("?api_key=pilot-secret");
  });

  it("does not reconnect after a 4401 close", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", MockWebSocket);

    const { useSupervisorWS } = await import("../useSupervisorWS");
    renderHook(() => useSupervisorWS());

    act(() => {
      MockWebSocket.instances[0]?.emitClose(4401);
    });
    act(() => {
      vi.advanceTimersByTime(3000);
    });

    expect(MockWebSocket.instances).toHaveLength(1);
  });

  it("upserts takeover alerts by session id", async () => {
    vi.stubGlobal("WebSocket", MockWebSocket);

    const { useSupervisorWS } = await import("../useSupervisorWS");
    const { result } = renderHook(() => useSupervisorWS());

    act(() => {
      MockWebSocket.instances[0]?.emitMessage({
        type: "takeover_alert",
        session_id: "session-1",
        reason: "first",
        timestamp: "2026-04-23T10:00:00Z",
      });
      MockWebSocket.instances[0]?.emitMessage({
        type: "takeover_alert",
        session_id: "session-1",
        reason: "updated",
        timestamp: "2026-04-23T10:01:00Z",
      });
    });

    expect(result.current.alerts).toHaveLength(1);
    expect(result.current.alerts[0]?.reason).toBe("updated");
  });
});
