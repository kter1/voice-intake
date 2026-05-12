import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { fetchSessions } from "../api";
import type { SessionSummary } from "../api";
import { useSupervisorWS } from "../hooks/useSupervisorWS";

const STATE_BADGE: Record<string, string> = {
  human_takeover: "bg-red-100 text-red-700",
  emergency_exit: "bg-orange-100 text-orange-700",
  consent_declined: "bg-yellow-100 text-yellow-700",
  close: "bg-green-100 text-green-700",
};

function StateBadge({ state }: { state: string }) {
  const cls = STATE_BADGE[state] ?? "bg-blue-100 text-blue-700";
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-mono ${cls}`}>{state}</span>
  );
}

export function Dashboard() {
  const { data: sessions = [], isLoading, error, refetch } = useQuery({
    queryKey: ["sessions"],
    queryFn: fetchSessions,
    refetchInterval: 5000,
  });

  const { alerts, dismissAlert } = useSupervisorWS();

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-800">Voice Intake - Supervisor Dashboard</h1>
        <div className="flex items-center gap-4">
          <Link to="/demo" className="text-sm text-blue-600 hover:underline font-medium">
            Try Demo
          </Link>
          <button
            onClick={() => refetch()}
            className="text-sm text-gray-500 hover:underline"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Takeover alerts */}
      {alerts.length > 0 && (
        <div className="space-y-2">
          {alerts.map((a) => (
            <div
              key={a.session_id + a.timestamp}
              className="flex items-center justify-between bg-red-50 border border-red-300 rounded-lg px-4 py-3"
            >
              <div>
                <span className="font-semibold text-red-700">🚨 Human takeover required</span>
                <span className="text-sm text-red-600 ml-2">
                  Session{" "}
                  <Link to={`/session/${a.session_id}`} className="underline font-mono">
                    {a.session_id.slice(0, 8)}…
                  </Link>{" "}
                  - {a.reason}
                </span>
              </div>
              <button
                onClick={() => dismissAlert(a.session_id)}
                className="text-xs text-red-500 hover:text-red-700 ml-4"
              >
                Acknowledge
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Sessions table */}
      {isLoading ? (
        <p className="text-gray-500">Loading sessions…</p>
      ) : error ? (
        <p className="text-red-600">Failed to load sessions.</p>
      ) : sessions.length === 0 ? (
        <p className="text-gray-400 italic">No sessions yet.</p>
      ) : (
        <div className="bg-white border rounded-lg overflow-hidden shadow-sm">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b">
              <tr>
                {["Session ID", "Operator", "State", "Started", "Disposition"].map((h) => (
                  <th key={h} className="px-4 py-3 text-left text-gray-600 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sessions.map((s: SessionSummary) => (
                <tr key={s.session_id} className="border-t hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-2.5">
                    <Link
                      to={`/session/${s.session_id}`}
                      className="font-mono text-blue-600 hover:underline"
                    >
                      {s.session_id.slice(0, 12)}…
                    </Link>
                  </td>
                  <td className="px-4 py-2.5">{s.operator_id}</td>
                  <td className="px-4 py-2.5">
                    <StateBadge state={s.current_state} />
                  </td>
                  <td className="px-4 py-2.5 text-gray-500 text-xs">
                    {new Date(s.started_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-2.5 text-gray-500 text-xs">{s.disposition ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
