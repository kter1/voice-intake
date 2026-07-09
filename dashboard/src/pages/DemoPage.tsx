/**
 * DemoPage - interactive guided demo of the voice intake system.
 *
 * - Auto-opens a session on mount
 * - Accepts speech via the browser microphone (Chrome/Edge) or typed text (all browsers)
 * - Renders AI responses by substituting {{variable}} placeholders in template content
 * - Shows the current state via StateIndicator
 * - Shows a live appointment-request panel (refreshed after each turn)
 * - "Reset" starts a fresh session
 */

import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { getAppointment, getCapabilities, openSession, submitTurn } from "../api";
import type { AppointmentRequest, Capabilities, SessionSummary, TurnResponse } from "../api";
import { StateIndicator } from "../components/StateIndicator";
import { VoiceInputPanel } from "../components/VoiceInputPanel";

// Frontend template content - must match DEFAULT_TEMPLATE_BUNDLE in templates.py.
// All backend-approved templates that may be emitted in demo or demo-local mode
// must appear here; missing entries render as [template_id] in the UI.
export const TEMPLATE_CONTENT: Record<string, string> = {
  // Core / consent / state-machine templates
  opening_disclosure: "You're speaking with an AI assistant. How can I help?",
  demo_capabilities_prompt:
    "I can help with a demo appointment intake, collect scheduling details, answer configured practice-policy questions when available, and route emergencies or staff handoff when needed. To start scheduling, tell me the reason for the visit.",
  recording_consent_prompt: "Do you consent to call recording where required by law?",
  ai_assistance_consent_prompt:
    "Do you consent to speaking with an AI assistant supervised by staff?",
  collect_field_prompt: "Please tell me your {{field_label}}.",
  readback_confirmation:
    "I have {{field_label}} ending in {{masked_value}}. Is that correct?",
  identity_threshold_denial: "I cannot continue until a staff member joins the call.",
  hold_message: "Please hold one moment.",
  emergency_redirect:
    "This sounds like an emergency. Please dial 911 now or seek emergency care " +
    "immediately. I'll stop the intake and alert staff.",
  closing_prompt: "Thank you. Your intake has been recorded for staff follow-up.",

  // Demo scheduling-flow templates
  appointment_reason_prompt: "What is the reason for the visit?",
  patient_name_prompt: "What is your full name?",
  dob_prompt: "What is your date of birth?",
  dob_clarification:
    "I didn't catch the date of birth. Please say it like month, day, year.",
  insurance_prompt: "What insurance should we use for this appointment?",
  insurance_in_network:
    "I'm showing {{insurance_name}} as in network. I can schedule the appointment. Is that okay?",
  insurance_out_of_network:
    "I'm showing {{insurance_name}} as out of network for this demo. " +
    "You may have higher out-of-pocket costs. Do you still want to continue?",
  insurance_unknown:
    "I can't verify {{insurance_name}} in this demo. " +
    "I can still schedule the request for staff review. Is that okay?",
  insurance_not_provided:
    "I can continue without insurance, but staff may need to review the request. Is that okay?",
  appointment_scheduled:
    "Your appointment has been scheduled. " +
    "A staff member can review the intake summary if needed.",
  appointment_request_created:
    "Your appointment request has been created for staff review. " +
    "A staff member can review the intake summary if needed.",
  human_handoff: "Of course. I'll alert staff to continue from here.",
  cannot_schedule_without_identity:
    "I can't schedule without that information, but I can alert staff to continue.",
  field_explanation_for_reason:
    "Staff use this to understand what type of appointment you need. " +
    "What is the reason for the visit?",
  field_explanation_for_name:
    "Staff use this to match the correct patient record and prepare the appointment request. " +
    "What is your full name?",
  field_explanation_for_dob:
    "Staff use this to match the correct patient record and avoid mixing up patients with similar names. " +
    "What is your date of birth?",
  field_explanation_for_insurance:
    "Staff use this to check the demo network status and prepare the appointment request. " +
    "What insurance should we use for this appointment?",
};

function renderTemplate(templateId: string, variables: Record<string, string>): string {
  let text = TEMPLATE_CONTENT[templateId] ?? `[${templateId}]`;
  for (const [key, val] of Object.entries(variables)) {
    text = text.replaceAll(`{{${key}}}`, val);
  }
  return text;
}

interface Message {
  id: string;
  role: "caller" | "ai" | "system";
  text: string;
  state?: string;
}

const STATE_FINAL = new Set(["close", "consent_declined", "human_takeover", "emergency_exit"]);

// ── Appointment panel ─────────────────────────────────────────────────────────

function networkBadge(status: string) {
  const labels: Record<string, string> = {
    in_network: "in network",
    out_of_network: "out of network",
    unknown: "unknown",
    not_provided: "self-pay / not provided",
  };
  const colours: Record<string, string> = {
    in_network: "text-green-700 bg-green-50",
    out_of_network: "text-amber-700 bg-amber-50",
    unknown: "text-gray-500 bg-gray-50",
    not_provided: "text-blue-700 bg-blue-50",
  };
  return (
    <span
      className={`text-xs font-medium px-1.5 py-0.5 rounded ${colours[status] ?? "text-gray-500 bg-gray-50"}`}
    >
      {labels[status] ?? status}
    </span>
  );
}

function statusBadge(status: string) {
  const labels: Record<string, string> = {
    collecting: "collecting",
    scheduled: "scheduled",
    staff_review: "staff review",
    emergency_redirect: "emergency",
    human_takeover: "staff handoff",
  };
  const colours: Record<string, string> = {
    collecting: "text-gray-500 bg-gray-100",
    scheduled: "text-green-700 bg-green-100",
    staff_review: "text-amber-700 bg-amber-100",
    emergency_redirect: "text-red-700 bg-red-100",
    human_takeover: "text-purple-700 bg-purple-100",
  };
  return (
    <span
      className={`text-xs font-semibold px-1.5 py-0.5 rounded ${colours[status] ?? "text-gray-500 bg-gray-100"}`}
    >
      {labels[status] ?? status}
    </span>
  );
}

function AppointmentPanel({ appt }: { appt: AppointmentRequest | null }) {
  const dash = <span className="text-gray-300">-</span>;
  const rows: { label: string; value: React.ReactNode }[] = [
    { label: "Status", value: appt ? statusBadge(appt.appointment_status) : dash },
    { label: "Reason", value: appt?.reason_for_visit ?? dash },
    { label: "Name", value: appt?.patient_name ?? dash },
    { label: "DOB", value: appt?.date_of_birth ?? dash },
    {
      label: "Insurance",
      value: appt?.insurance_name ?? dash,
    },
    {
      label: "Network",
      value: appt ? networkBadge(appt.insurance_network_status) : dash,
    },
    {
      label: "Scheduled slot",
      value: appt?.scheduled_slot
        ? <span className="font-mono text-xs">{appt.scheduled_slot}</span>
        : dash,
    },
  ];

  return (
    <div className="bg-white border rounded-xl p-4 shadow-sm">
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
        Appointment Request
      </h3>
      <dl className="space-y-1.5">
        {rows.map(({ label, value }) => (
          <div key={label} className="flex justify-between gap-2 text-xs">
            <dt className="text-gray-400 shrink-0">{label}</dt>
            <dd className="text-gray-800 text-right">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function DemoPage() {
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [appointment, setAppointment] = useState<AppointmentRequest | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const hasStarted = useRef(false);
  const sessionRequestId = useRef(0);

  // Fetch capabilities once on mount (for forward-compat info display)
  useEffect(() => {
    getCapabilities().then(setCapabilities).catch(() => {/* non-critical */});
  }, []);

  useEffect(() => {
    if (hasStarted.current) return;
    hasStarted.current = true;
    startSession();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function startSession() {
    const requestId = ++sessionRequestId.current;
    setLoading(true);
    setError(null);
    setSession(null);
    setMessages([]);
    setAppointment(null);
    try {
      const s = await openSession("demo");
      if (requestId !== sessionRequestId.current) return;
      setSession(s);
      // Show the opening AI message
      addMessage({
        role: "ai",
        text: renderTemplate("opening_disclosure", {}),
        state: s.current_state,
      });
    } catch (e) {
      if (requestId !== sessionRequestId.current) return;
      setError(e instanceof Error ? e.message : "Failed to open session");
    } finally {
      if (requestId === sessionRequestId.current) {
        setLoading(false);
      }
    }
  }

  function addMessage(msg: Omit<Message, "id">) {
    setMessages((prev) => [...prev, { ...msg, id: `${Date.now()}-${Math.random()}` }]);
  }

  async function refreshAppointment(sessionId: string) {
    try {
      const appt = await getAppointment(sessionId);
      setAppointment(appt);
    } catch {
      // Non-critical; appointment panel just stays stale
    }
  }

  async function handleTranscript(transcript: string) {
    if (!session) return;
    setLoading(true);
    setError(null);

    addMessage({ role: "caller", text: transcript });

    try {
      const result: TurnResponse = await submitTurn(session.session_id, transcript);

      // Update session state
      setSession((prev) => prev ? { ...prev, current_state: result.current_state } : prev);

      if (result.voice_action) {
        // Guard-approved natural phrasing wins; template content is the fallback.
        const aiText = result.voice_action.spoken_text ||
          renderTemplate(
            result.voice_action.template_id,
            result.voice_action.allowed_variables,
          );
        addMessage({ role: "ai", text: aiText, state: result.current_state });
      } else if (result.rejection) {
        // Provide humane rejection message instead of raw technical copy
        const rejectionMessages: Record<string, string> = {
          llm_timeout: "I'm having trouble responding quickly. Please try again or ask to speak with staff.",
          disallowed_transition: "That request can't be processed right now. Please try something else.",
          invalid_template_id: "I didn't understand that properly. Please try again.",
          pii_in_variables: "I can't process that information that way. Please try again.",
          injection_marker: "I can't process that. Please try again.",
        };
        const rejectionText = rejectionMessages[result.rejection.reason] ||
          `Response not accepted. Please try again.`;
        addMessage({
          role: "system",
          text: rejectionText,
          state: result.current_state,
        });
      }

      // Refresh appointment panel after every turn
      await refreshAppointment(session.session_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const isFinal = session ? STATE_FINAL.has(session.current_state) : false;

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b px-6 py-3 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <Link to="/" className="text-blue-600 hover:underline text-sm">
            ← Supervisor view
          </Link>
          <h1 className="text-lg font-semibold text-gray-800">Voice Intake Demo</h1>
        </div>
        <button
          onClick={startSession}
          disabled={loading}
          className="text-sm text-gray-500 hover:text-gray-700 border rounded px-3 py-1 disabled:opacity-40"
        >
          Reset
        </button>
      </div>

      <div className="flex flex-1 max-w-4xl mx-auto w-full gap-4 p-4">
        {/* Chat area */}
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex-1 overflow-y-auto space-y-3 pb-4">
            {messages.length === 0 && !loading && (
              <div className="text-center text-gray-400 italic mt-12">
                {error ? (
                  <p className="text-red-500">{error}</p>
                ) : (
                  <p>Starting session…</p>
                )}
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.role === "caller" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={[
                    "max-w-xs lg:max-w-md px-4 py-2.5 rounded-2xl text-sm shadow-sm",
                    msg.role === "caller"
                      ? "bg-blue-600 text-white rounded-br-sm"
                      : msg.role === "system"
                      ? "bg-yellow-50 border border-yellow-200 text-yellow-800 rounded-bl-sm"
                      : "bg-white border text-gray-800 rounded-bl-sm",
                  ].join(" ")}
                >
                  <p>{msg.text}</p>
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="bg-white border rounded-2xl rounded-bl-sm px-4 py-2.5 text-sm text-gray-400 shadow-sm">
                  Thinking…
                </div>
              </div>
            )}

            {error && messages.length > 0 && (
              <p className="text-center text-sm text-red-500">{error}</p>
            )}

            <div ref={bottomRef} />
          </div>

          {/* Input */}
          {!isFinal ? (
            <div className="border-t bg-white rounded-xl p-3 shadow-sm">
              <VoiceInputPanel
                onTranscript={handleTranscript}
                disabled={loading || !session}
              />
            </div>
          ) : (
            <div className="border-t bg-green-50 rounded-xl p-4 text-center">
              <p className="text-green-700 font-medium text-sm">
                Session ended - state: <span className="font-mono">{session?.current_state}</span>
              </p>
              <button
                onClick={startSession}
                className="mt-2 text-sm text-blue-600 hover:underline"
              >
                Start a new session
              </button>
            </div>
          )}
        </div>

        {/* Right panel - state + appointment */}
        <div className="hidden md:flex flex-col w-56 shrink-0 gap-4">
          <div className="bg-white border rounded-xl p-4 shadow-sm sticky top-4">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
              State Machine
            </h3>
            {session ? (
              <StateIndicator currentState={session.current_state} />
            ) : (
              <p className="text-xs text-gray-400 italic">No session yet</p>
            )}
            {session && (
              <p className="mt-3 text-xs text-gray-400">
                Session{" "}
                <span className="font-mono">{session.session_id.slice(0, 8)}…</span>
              </p>
            )}
            {capabilities?.backend_asr_configured && (
              <p className="mt-2 text-xs text-green-600">Backend ASR configured</p>
            )}
          </div>

          <AppointmentPanel appt={appointment} />
        </div>
      </div>
    </div>
  );
}
