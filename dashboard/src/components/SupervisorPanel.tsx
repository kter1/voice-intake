import { useState } from "react";
import { supervisorIntervene } from "../api";

interface Props {
  sessionId: string;
  onSuccess: () => void;
}

export function SupervisorPanel({ sessionId, onSuccess }: Props) {
  const [reasonCode, setReasonCode] = useState("");
  const [notes, setNotes] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [confirming, setConfirming] = useState(false);

  const handleTakeover = async () => {
    if (!reasonCode.trim()) {
      setError("Reason code is required.");
      return;
    }
    setConfirming(true);
  };

  const confirmTakeover = async () => {
    setLoading(true);
    setError(null);
    try {
      await supervisorIntervene(sessionId, "forced_takeover", reasonCode, notes);
      setConfirming(false);
      onSuccess();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="border rounded-lg p-4 bg-white shadow-sm">
      <h3 className="font-semibold text-gray-700 mb-3">Supervisor Actions</h3>

      <label className="block text-sm font-medium text-gray-600 mb-1">
        Reason Code <span className="text-red-500">*</span>
      </label>
      <input
        className="w-full border rounded px-3 py-1.5 text-sm mb-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
        placeholder="e.g. caller_distressed"
        value={reasonCode}
        onChange={(e) => setReasonCode(e.target.value)}
        disabled={confirming}
      />

      <label className="block text-sm font-medium text-gray-600 mb-1">Notes (optional)</label>
      <textarea
        className="w-full border rounded px-3 py-1.5 text-sm mb-3 focus:outline-none focus:ring-2 focus:ring-blue-400"
        rows={2}
        placeholder="Describe the situation…"
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        disabled={confirming}
      />

      {error && (
        <p className="text-red-600 text-sm mb-2">{error}</p>
      )}

      {!confirming ? (
        <button
          onClick={handleTakeover}
          className="bg-red-600 hover:bg-red-700 text-white text-sm font-medium px-4 py-2 rounded transition-colors"
        >
          Force Takeover
        </button>
      ) : (
        <div className="flex gap-2 items-center">
          <p className="text-sm text-gray-700">Are you sure? This cannot be undone.</p>
          <button
            onClick={confirmTakeover}
            disabled={loading}
            className="bg-red-700 hover:bg-red-800 text-white text-sm font-medium px-3 py-1.5 rounded"
          >
            {loading ? "…" : "Confirm"}
          </button>
          <button
            onClick={() => setConfirming(false)}
            className="bg-gray-200 hover:bg-gray-300 text-gray-700 text-sm font-medium px-3 py-1.5 rounded"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}
