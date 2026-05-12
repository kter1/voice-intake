import { afterEach, describe, expect, it, vi } from "vitest";

describe("api auth headers", () => {
  afterEach(() => {
    vi.resetModules();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("sends X-API-Key when configured", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ sessions: [] }),
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubEnv("VITE_API_KEY", "pilot-secret");

    const { fetchSessions } = await import("../api");
    await fetchSessions();

    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/sessions", {
      headers: { "X-API-Key": "pilot-secret" },
    });
  });
});
