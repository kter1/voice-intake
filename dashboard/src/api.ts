const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY ?? "";
const authHeaders: Record<string, string> = API_KEY ? { "X-API-Key": API_KEY } : {};

export interface TurnResponse {
  session_id: string;
  turn_id: string;
  current_state: string;
  voice_action: {
    action_type: string;
    template_id: string;
    allowed_variables: Record<string, string>;
    interruptible: boolean;
    timeout_ms: number;
  } | null;
  rejection: { reason: string } | null;
}

export interface SessionSummary {
  session_id: string;
  current_state: string;
  mode: string;
  started_at: string;
  operator_id: string;
  ended_at: string | null;
  disposition: string | null;
}

export interface AuditEvent {
  event_id: string;
  event_type: string;
  actor: string;
  timestamp: string;
  template_id: string | null;
  before_after_hash: string;
  chain_hash: string;
}

export interface AuditResponse {
  session_id: string;
  events: AuditEvent[];
}

export const fetchSessions = async (): Promise<SessionSummary[]> => {
  const r = await fetch(`${BASE}/sessions`, { headers: authHeaders });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await r.json();
  return data.sessions as SessionSummary[];
};

export const fetchSession = async (id: string): Promise<SessionSummary> => {
  const r = await fetch(`${BASE}/session/${id}`, { headers: authHeaders });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
};

export const fetchAudit = async (id: string): Promise<AuditResponse> => {
  const r = await fetch(`${BASE}/session/${id}/audit`, { headers: authHeaders });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
};

export const openSession = async (operatorId = "demo"): Promise<SessionSummary> => {
  const r = await fetch(`${BASE}/session/open`, {
    method: "POST",
    headers: { ...authHeaders, "Content-Type": "application/json" },
    body: JSON.stringify({
      telephony_call_id: `browser-${Date.now()}`,
      operator_id: operatorId,
      stir_shaken_attestation: "unknown",
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${r.status}`);
  }
  return r.json();
};

export const submitTurn = async (
  sessionId: string,
  transcript: string,
  confidence = 1.0,
): Promise<TurnResponse> => {
  const r = await fetch(`${BASE}/session/${sessionId}/turn`, {
    method: "POST",
    headers: { ...authHeaders, "Content-Type": "application/json" },
    body: JSON.stringify({ transcript, asr_confidence: confidence }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${r.status}`);
  }
  return r.json();
};

export interface Capabilities {
  mock_asr: boolean;
  mock_llm: boolean;
  mock_rag: boolean;
  demo_router: boolean;
  backend_asr_configured: boolean;
  voice_modes: ("browser_speech" | "text" | "backend_asr")[];
}

export interface AppointmentRequest {
  appointment_id: string;
  session_id: string;
  reason_for_visit: string | null;
  patient_name: string | null;
  date_of_birth: string | null;
  insurance_name: string | null;
  insurance_network_status: "in_network" | "out_of_network" | "unknown" | "not_provided";
  appointment_status: string;
  scheduled_slot: string | null;
  created_at: string;
  updated_at: string;
}

export const getCapabilities = async (): Promise<Capabilities> => {
  const r = await fetch(`${BASE}/capabilities`, { headers: authHeaders });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
};

export const getAppointment = async (
  sessionId: string,
): Promise<AppointmentRequest | null> => {
  const r = await fetch(`${BASE}/session/${sessionId}/appointment`, {
    headers: authHeaders,
  });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
};

export const supervisorIntervene = async (
  sessionId: string,
  interventionType: string,
  reasonCode: string,
  notes?: string
): Promise<{ session_id: string; current_state: string }> => {
  const r = await fetch(`${BASE}/session/${sessionId}/supervisor/intervene`, {
    method: "POST",
    headers: { ...authHeaders, "Content-Type": "application/json" },
    body: JSON.stringify({
      intervention_type: interventionType,
      reason_code: reasonCode,
      notes: notes ?? "",
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail ?? `HTTP ${r.status}`);
  }
  return r.json();
};
