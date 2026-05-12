import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { fetchSession, fetchAudit } from "../api";
import { StateIndicator } from "../components/StateIndicator";
import { SupervisorPanel } from "../components/SupervisorPanel";

export function SessionPage() {
  const { id } = useParams<{ id: string }>();
  const qc = useQueryClient();
  const [showAudit, setShowAudit] = useState(false);

  const { data: session, isLoading, error } = useQuery({
    queryKey: ["session", id],
    queryFn: () => fetchSession(id!),
    refetchInterval: 5000,
    enabled: !!id,
  });

  const { data: audit } = useQuery({
    queryKey: ["audit", id],
    queryFn: () => fetchAudit(id!),
    enabled: showAudit && !!id,
  });

  if (isLoading) return <div className="p-8 text-gray-500">Loading…</div>;
  if (error || !session)
    return <div className="p-8 text-red-600">Session not found.</div>;

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <Link to="/" className="text-blue-600 hover:underline text-sm">← Dashboard</Link>
        <h1 className="text-xl font-bold text-gray-800 font-mono">{session.session_id}</h1>
      </div>

      {/* Session info */}
      <div className="grid grid-cols-2 gap-4 bg-white border rounded-lg p-4 shadow-sm text-sm">
        <Info label="Operator" value={session.operator_id} />
        <Info label="Mode" value={session.mode} />
        <Info label="Started" value={new Date(session.started_at).toLocaleString()} />
        <Info label="Ended" value={session.ended_at ? new Date(session.ended_at).toLocaleString() : "-"} />
        <Info label="Disposition" value={session.disposition ?? "-"} />
      </div>

      {/* State machine */}
      <div className="bg-white border rounded-lg p-4 shadow-sm">
        <h2 className="font-semibold text-gray-700 mb-3">Current State</h2>
        <StateIndicator currentState={session.current_state} />
      </div>

      {/* Supervisor actions (only show if not already taken over or closed) */}
      {!["human_takeover", "close", "consent_declined"].includes(session.current_state) && (
        <SupervisorPanel
          sessionId={session.session_id}
          onSuccess={() => qc.invalidateQueries({ queryKey: ["session", id] })}
        />
      )}

      {/* Audit trail toggle */}
      <div>
        <button
          onClick={() => setShowAudit((v) => !v)}
          className="text-sm text-blue-600 hover:underline"
        >
          {showAudit ? "Hide" : "Show"} audit trail
        </button>
        {showAudit && audit && (
          <div className="mt-3 border rounded-lg overflow-hidden">
            <table className="w-full text-xs">
              <thead className="bg-gray-50">
                <tr>
                  {["Time", "Type", "Actor", "Template"].map((h) => (
                    <th key={h} className="px-3 py-2 text-left text-gray-600 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {audit.events.map((e) => (
                  <tr key={e.event_id} className="border-t hover:bg-gray-50">
                    <td className="px-3 py-1.5 text-gray-500 font-mono">
                      {new Date(e.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="px-3 py-1.5 font-mono">{e.event_type}</td>
                    <td className="px-3 py-1.5">{e.actor}</td>
                    <td className="px-3 py-1.5 font-mono text-gray-500">{e.template_id ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-gray-500">{label}:</span>{" "}
      <span className="font-medium text-gray-800">{value}</span>
    </div>
  );
}
