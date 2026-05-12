import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

let latestRecognition: MockSpeechRecognition | null = null;

class MockSpeechRecognition {
  lang = "";
  interimResults = false;
  continuous = false;
  onresult: ((event: unknown) => void) | null = null;
  onerror: (() => void) | null = null;
  onend: (() => void) | null = null;
  start = vi.fn();
  stop = vi.fn(() => {
    this.onend?.();
  });
  abort = vi.fn();

  constructor() {
    latestRecognition = this;
  }
}

function getMicButton(): HTMLButtonElement {
  const micButton = screen
    .getAllByRole("button")
    .find((element) => element.hasAttribute("aria-pressed"));
  if (!(micButton instanceof HTMLButtonElement)) {
    throw new Error("Mic button not found");
  }
  return micButton;
}

async function loadVoiceInputPanel() {
  const module = await import("../VoiceInputPanel");
  return module.VoiceInputPanel;
}

describe("VoiceInputPanel", () => {
  beforeEach(() => {
    vi.resetModules();
    latestRecognition = null;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    delete (window as Window & { SpeechRecognition?: typeof MockSpeechRecognition }).SpeechRecognition;
    delete (window as Window & { webkitSpeechRecognition?: typeof MockSpeechRecognition })
      .webkitSpeechRecognition;
  });

  it('switches to textarea on "type instead"', async () => {
    Object.defineProperty(window, "SpeechRecognition", {
      configurable: true,
      value: MockSpeechRecognition,
    });
    const VoiceInputPanel = await loadVoiceInputPanel();

    render(<VoiceInputPanel onTranscript={vi.fn()} />);

    await userEvent.click(screen.getByRole("button", { name: /type instead/i }));

    expect(screen.getByRole("textbox")).toBeTruthy();
  });

  it("returns to microphone mode from text mode", async () => {
    Object.defineProperty(window, "SpeechRecognition", {
      configurable: true,
      value: MockSpeechRecognition,
    });
    const VoiceInputPanel = await loadVoiceInputPanel();

    render(<VoiceInputPanel onTranscript={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /type instead/i }));

    expect(screen.getByRole("textbox")).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: /use microphone/i }));

    expect(screen.queryByRole("textbox")).toBeNull();
    expect(getMicButton().getAttribute("aria-pressed")).toBe("false");
  });

  it("toggles mic listening state on click", async () => {
    Object.defineProperty(window, "SpeechRecognition", {
      configurable: true,
      value: MockSpeechRecognition,
    });
    const VoiceInputPanel = await loadVoiceInputPanel();

    render(<VoiceInputPanel onTranscript={vi.fn()} />);

    const micButton = getMicButton();
    await userEvent.click(micButton);
    expect(micButton.getAttribute("aria-pressed")).toBe("true");
    expect(latestRecognition?.continuous).toBe(true);

    await userEvent.click(micButton);
    expect(micButton.getAttribute("aria-pressed")).toBe("false");
  });

  it("submits accumulated final transcript only after explicit stop", async () => {
    Object.defineProperty(window, "SpeechRecognition", {
      configurable: true,
      value: MockSpeechRecognition,
    });
    const onTranscript = vi.fn();
    const VoiceInputPanel = await loadVoiceInputPanel();

    render(<VoiceInputPanel onTranscript={onTranscript} />);

    const micButton = getMicButton();
    await userEvent.click(micButton);

    const finalOne = Object.assign([{ transcript: "Hello there" }], { isFinal: true });
    const finalTwo = Object.assign([{ transcript: "general kenobi" }], { isFinal: true });
    latestRecognition?.onresult?.({
      resultIndex: 0,
      results: [finalOne, finalTwo],
    });

    expect(onTranscript).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(
        screen.getByText((content) => content.includes("Hello there general kenobi")),
      ).toBeTruthy();
    });

    await userEvent.click(micButton);

    expect(latestRecognition?.stop).toHaveBeenCalledTimes(1);
    expect(onTranscript).toHaveBeenCalledTimes(1);
    expect(onTranscript).toHaveBeenCalledWith("Hello there general kenobi");
  });

  it("shows a microphone error without submitting", async () => {
    Object.defineProperty(window, "SpeechRecognition", {
      configurable: true,
      value: MockSpeechRecognition,
    });
    const onTranscript = vi.fn();
    const VoiceInputPanel = await loadVoiceInputPanel();

    render(<VoiceInputPanel onTranscript={onTranscript} />);

    await userEvent.click(getMicButton());
    latestRecognition?.onerror?.();

    expect(onTranscript).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(screen.getByText(/microphone unavailable/i)).toBeTruthy();
    });
  });

  it("submits on Enter but not on Shift+Enter", async () => {
    const onTranscript = vi.fn();
    const VoiceInputPanel = await loadVoiceInputPanel();

    render(<VoiceInputPanel onTranscript={onTranscript} />);

    const textbox = screen.getByRole("textbox");
    await userEvent.type(textbox, "Need help");

    fireEvent.keyDown(textbox, { key: "Enter", shiftKey: true });
    expect(onTranscript).not.toHaveBeenCalled();

    fireEvent.keyDown(textbox, { key: "Enter" });
    expect(onTranscript).toHaveBeenCalledTimes(1);
    expect(onTranscript).toHaveBeenCalledWith("Need help");
  });
});
