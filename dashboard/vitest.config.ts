import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// Vitest config is intentionally separate from vite.config.ts because vitest 2.x
// bundles vite 5.x types for its `test` field augmentation, which conflicts with
// the installed vite 8.x. Vitest auto-discovers this file ahead of vite.config.ts.

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    restoreMocks: true,
  },
})
