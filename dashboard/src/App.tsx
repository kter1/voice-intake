import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Dashboard } from "./pages/Dashboard";
import { DemoPage } from "./pages/DemoPage";
import { SessionPage } from "./pages/SessionPage";
import "./index.css";

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1 } } });

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <div className="min-h-screen bg-gray-50">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/demo" element={<DemoPage />} />
            <Route path="/session/:id" element={<SessionPage />} />
          </Routes>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
