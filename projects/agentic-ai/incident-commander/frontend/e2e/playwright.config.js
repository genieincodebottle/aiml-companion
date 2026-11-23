// End-to-end tests: the real backend + the real console, driven in a browser.
//
//   npm run test:e2e        (from frontend/; needs `uv sync` done in backend/)
//
// Both servers are started here on dedicated ports (8765, 5765), so the tests
// never reuse your dev servers. The backend is forced into offline mode, so the
// suite needs no API key and never spends model quota, even if one is configured.
import { defineConfig, devices } from '@playwright/test';
import os from 'node:os';
import path from 'node:path';

const FRONTEND_DIR = path.resolve(import.meta.dirname, '..');
const BACKEND_DIR = path.resolve(import.meta.dirname, '../../backend');
const BACKEND = 'http://localhost:8765';
const CONSOLE = 'http://localhost:5765';

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
  use: {
    baseURL: CONSOLE,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command: 'uv run uvicorn main:app --port 8765',
      cwd: BACKEND_DIR,
      url: `${BACKEND}/api/health`,
      reuseExistingServer: false,
      timeout: 120_000,
      env: {
        IC_LLM_MODE: 'offline',
        IC_OFFLINE_STEP_DELAY_S: '0.05',
        IC_DATABASE_URL: '',
        IC_AUTH_REQUIRED: 'false',
        IC_RATE_LIMIT_PER_MIN: '0',
        IC_CORS_ORIGINS: CONSOLE,
        IC_MEMORY_PATH: path.join(os.tmpdir(), `ic-e2e-memory-${process.pid}.json`),
      },
    },
    {
      command: 'npm run dev -- --port 5765 --strictPort',
      cwd: FRONTEND_DIR,
      url: CONSOLE,
      reuseExistingServer: false,
      timeout: 120_000,
      env: { IC_BACKEND_URL: BACKEND },
    },
  ],
});
