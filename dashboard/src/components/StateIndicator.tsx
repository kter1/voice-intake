/**
 * Visual state-machine indicator.
 * Highlights the current state node; all other states are shown as grey pills.
 */
const ALL_STATES = [
  "opening",
  "consent_recording",
  "consent_ai_assistance",
  "consent_declined",
  "identity_capture",
  "demographics",
  "insurance",
  "reason_for_visit",
  "readback",
  "disposition",
  "close",
  "human_takeover",
  "emergency_exit",
  "manual_mode",
];

const STATE_COLORS: Record<string, string> = {
  human_takeover: "bg-red-500 text-white",
  emergency_exit: "bg-orange-500 text-white",
  consent_declined: "bg-yellow-500 text-white",
  close: "bg-green-500 text-white",
};

interface Props {
  currentState: string;
}

export function StateIndicator({ currentState }: Props) {
  return (
    <div className="flex flex-wrap gap-1">
      {ALL_STATES.map((s) => {
        const isActive = s === currentState;
        const color = isActive
          ? (STATE_COLORS[s] ?? "bg-blue-600 text-white")
          : "bg-gray-100 text-gray-500";
        return (
          <span
            key={s}
            className={`px-2 py-0.5 rounded text-xs font-mono ${color} ${
              isActive ? "ring-2 ring-offset-1 ring-blue-400" : ""
            }`}
          >
            {s}
          </span>
        );
      })}
    </div>
  );
}
