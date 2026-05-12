import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Test config (vitest) lives in vitest.config.ts. Splitting avoids the type
// collision between the installed vite (8.x) and the older vite (5.x) that
// vitest 2.x bundles for its own type augmentation.

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/session": "http://localhost:8000",
      "/sessions": "http://localhost:8000",
      "/knowledge": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
  build: { outDir: "dist" },
})
