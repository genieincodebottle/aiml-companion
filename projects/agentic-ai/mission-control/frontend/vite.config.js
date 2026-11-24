import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The backend the dev server proxies /api to. The e2e suite points this at its own
// backend on a separate port so it never touches your dev servers.
const backend = process.env.MC_BACKEND_URL || 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: backend, changeOrigin: true },
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.js'],
  },
});
