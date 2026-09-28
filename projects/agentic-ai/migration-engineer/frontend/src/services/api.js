import axios from 'axios';

// In dev, VITE_API_URL is empty so requests go through the Vite proxy
// (/api -> http://localhost:8000). In other environments, point it at the backend.
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// System / health
export const systemAPI = {
  getHealth: async () => {
    const response = await api.get('/api/health');
    return response.data;
  },
};

// Jobs (migration catalog)
export const jobAPI = {
  getJobs: async () => {
    const response = await api.get('/api/jobs');
    return response.data;
  },
};

// Runs (migration executions)
export const runAPI = {
  // body: { job_id, auto_approve? } -> { run_id }
  triggerRun: async ({ jobId, autoApprove }) => {
    const body = { job_id: jobId };
    if (typeof autoApprove === 'boolean') body.auto_approve = autoApprove;
    const response = await api.post('/api/runs', body);
    return response.data;
  },

  listRuns: async () => {
    const response = await api.get('/api/runs');
    return response.data;
  },

  getRun: async (runId) => {
    const response = await api.get(`/api/runs/${runId}`);
    return response.data;
  },

  // body: { repo_id, decision: 'approve' | 'reject', note? } -> { ok: true }
  // 409 if there is no pending approval for that repo_id.
  decide: async (runId, { repoId, decision, note }) => {
    const body = { repo_id: repoId, decision };
    if (note) body.note = note;
    const response = await api.post(`/api/runs/${runId}/approve`, body);
    return response.data;
  },
};

/**
 * Subscribe to the live SSE stream for a migration run.
 * Wraps the native browser EventSource.
 *
 * @param {string} runId
 * @param {(event: object) => void} onEvent - called with each parsed JSON event
 * @param {{ onError?: (err) => void, onOpen?: () => void }} [opts]
 * @returns {() => void} unsubscribe function that closes the stream
 */
export function subscribeToRun(runId, onEvent, opts = {}) {
  const base = API_BASE_URL || '';
  const url = `${base}/api/runs/${runId}/stream`;
  const source = new EventSource(url);

  source.onopen = () => {
    if (opts.onOpen) opts.onOpen();
  };

  source.onmessage = (e) => {
    if (!e.data) return;
    try {
      const parsed = JSON.parse(e.data);
      onEvent(parsed);
    } catch (err) {
      // Ignore non-JSON keepalive frames, surface real parse issues.
      if (opts.onError) opts.onError(err);
    }
  };

  source.onerror = (err) => {
    if (opts.onError) opts.onError(err);
  };

  return () => {
    source.close();
  };
}

export default api;
