import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The proxy target is configurable because 8000 is a popular port: if something
// else on the machine already owns it, you can run this backend elsewhere with
// `IC_BACKEND_URL=http://localhost:8010 npm run dev` instead of editing this file.
const BACKEND = process.env.IC_BACKEND_URL || 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: BACKEND,
        changeOrigin: true,
      },
    },
  },
});
