import { useEffect, useRef, useState } from "react";

export interface TakeoverAlert {
  type: "takeover_alert";
  session_id: string;
  reason: string;
  timestamp: string;
}

const API_KEY = import.meta.env.VITE_API_KEY ?? "";
const WS_BASE =
  (import.meta.env.VITE_API_BASE ?? "http://localhost:8000")
    .replace(/^http/, "ws") + "/ws/supervisors";

export function useSupervisorWS() {
  const [alerts, setAlerts] = useState<TakeoverAlert[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    const connect = () => {
      const url = API_KEY ? `${WS_BASE}?api_key=${encodeURIComponent(API_KEY)}` : WS_BASE;
      ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data) as TakeoverAlert;
          if (msg.type === "takeover_alert") {
            setAlerts((prev) => [msg, ...prev.filter((item) => item.session_id !== msg.session_id)].slice(0, 20));
          }
        } catch {
          // ignore malformed messages
        }
      };

      ws.onclose = (event) => {
        if (event.code === 4401) return;
        // Reconnect after 3 s
        reconnectTimer = setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, []);

  const dismissAlert = (sessionId: string) => {
    setAlerts((prev) => prev.filter((a) => a.session_id !== sessionId));
  };

  return { alerts, dismissAlert };
}
