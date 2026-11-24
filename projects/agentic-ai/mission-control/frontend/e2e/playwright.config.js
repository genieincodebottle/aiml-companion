// End-to-end tests: the real backend + the real console, driven in a browser.
//
//   npm run test:e2e        (from frontend/; needs `uv sync` done in backend/)
//
// Both servers start here on their own ports (8865, 5865), so the suite never reuses
// your dev servers. The backend is forced offline, so it needs no key and never
// spends model quota even if one is configured.
import { defineConfig, devices } from '@playwright/test';
import os from 'node:os';
import path from 'node:path';

const FRONTEND_DIR = path.resolve(import.meta.dirname, '..');
const BACKEND_DIR = path.resolve(import.meta.dirname, '../../backend');
const BACKEND = 'http://localhost:8865';
const CONSOLE = 'http://localhost:5865';

export default defineConfig({
  testDir: '.',
  testMatch: '*.e2e.js', // not *.test.js, so Vitest never picks these up
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  workers: 1, // one backend, shared state: run serially
  reporter: process.env.CI
    ? [['list'], ['html', { open: 'never', outputFolder: '../playwright-report' }]]
    : 'list',
  outputDir: '../test-results',
  use: { baseURL: CONSOLE, trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command: 'uv run uvicorn main:app --port 8865',
      cwd: BACKEND_DIR,
      url: `${BACKEND}/api/health`,
      reuseExistingServer: false,
      timeout: 120_000,
      env: {
        MC_MODE: 'offline',
        MC_OPERATORS: '',
        MC_AUTO_APPROVE: '0',
        MC_CORS_ORIGINS: CONSOLE,
        MC_DATA_DIR: path.join(os.tmpdir(), `mc-e2e-${process.pid}`),
      },
    },
    {
      command: 'npm run dev -- --port 5865 --strictPort',
      cwd: FRONTEND_DIR,
      url: CONSOLE,
      reuseExistingServer: false,
      timeout: 120_000,
      env: { MC_BACKEND_URL: BACKEND },
    },
  ],
});
