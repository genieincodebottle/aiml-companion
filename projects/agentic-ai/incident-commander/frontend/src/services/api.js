import axios from 'axios';

// In dev, VITE_API_URL is empty so requests go through the Vite proxy
// (/api -> http://localhost:8000). In other environments, point it at the backend.
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// Optional shared secret, matching IC_API_KEY on the backend. Unset in local dev,
// where the backend leaves auth off.
const API_KEY = import.meta.env.VITE_API_KEY || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
  },
});

// --- session token ----------------------------------------------------------
// Kept in localStorage so a refresh does not log you out. That is a real
// trade-off rather than an oversight: localStorage is readable by any script on
// the origin, so an XSS becomes a token theft. An httpOnly cookie would close
// that at the cost of needing CSRF protection. Noted here because "we used
// localStorage" is a decision, not a default.
export const TOKEN_KEY = 'ic.token';

export function getToken() {
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token) {
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Private mode: the session still works, it just will not survive a reload.
  }
}

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// A 401 means the token is gone or expired. Drop it and let the router send the
// user to /login, rather than leaving a dead token to fail every later request.
// `window.location` is deliberately NOT touched here - a hard redirect from an
// interceptor loses in-flight React state and makes the login bounce jarring.
let onUnauthorized = null;
export function setUnauthorizedHandler(fn) {
  onUnauthorized = fn;
}

api.interceptors.response.use(
  (r) => r,
  (error) => {
    if (error?.response?.status === 401) {
      setToken(null);
      onUnauthorized?.();
    }
    return Promise.reject(error);
  }
);

// Auth
export const authAPI = {
  login: async (username, password) => {
    const response = await api.post('/api/auth/login', { username, password });
    return response.data;
  },

  me: async () => {
    const response = await api.get('/api/auth/me');
    return response.data;
  },

  register: async ({ username, password, email, role }) => {
    const response = await api.post('/api/auth/register', { username, password, email, role });
    return response.data;
  },

  logout: async () => {
    try {
      await api.post('/api/auth/logout');
    } finally {
      setToken(null);
    }
  },
};

// System / health
export const systemAPI = {
  getHealth: async () => {
    const response = await api.get('/api/health');
    return response.data;
  },
};

// Scenarios (fault catalog)
export const scenarioAPI = {
  getScenarios: async () => {
    const response = await api.get('/api/scenarios');
    return response.data;
  },
};

// Incidents
export const incidentAPI = {
  // body: { scenario_id, service?, auto_approve? } -> { incident_id }
  triggerIncident: async ({ scenarioId, service, autoApprove }) => {
    const body = { scenario_id: scenarioId };
    if (service) body.service = service;
    if (typeof autoApprove === 'boolean') body.auto_approve = autoApprove;
    const response = await api.post('/api/incidents', body);
    return response.data;
  },

  listIncidents: async () => {
    const response = await api.get('/api/incidents');
    return response.data;
  },

  // Aggregates, computed in Postgres when one is configured and in memory when not.
  getStats: async () => {
    const response = await api.get('/api/incidents/stats');
    return response.data;
  },

  getIncident: async (incidentId) => {
    const response = await api.get(`/api/incidents/${incidentId}`);
    return response.data;
  },

  // body: { action_id, decision: 'approve' | 'reject', note? } -> { ok: true }
  decideAction: async (incidentId, { actionId, decision, note }) => {
    const body = { action_id: actionId, decision };
    if (note) body.note = note;
    const response = await api.post(`/api/incidents/${incidentId}/approve`, body);
    return response.data;
  },

  // Ask a running incident to stop at its next checkpoint. Cooperative: the
  // incident still writes a postmortem.
  cancelIncident: async (incidentId) => {
    const response = await api.post(`/api/incidents/${incidentId}/cancel`);
    return response.data;
  },

  // Per-agent token / cost / latency breakdown for one incident.
  getUsage: async (incidentId) => {
    const response = await api.get(`/api/incidents/${incidentId}/usage`);
    return response.data;
  },

  // The loop's actual ceilings and safety settings.
  getConfig: async () => {
    const response = await api.get('/api/config');
    return response.data;
  },
};

// Runbook memory: what the agent has learned across incidents.
export const memoryAPI = {
  getMemory: async () => {
    const response = await api.get('/api/memory');
    return response.data;
  },

  search: async (query, topK = 3) => {
    const response = await api.get('/api/memory/search', { params: { q: query, top_k: topK } });
    return response.data;
  },
};

/**
 * Subscribe to the live SSE stream for an incident.
 * Wraps the native browser EventSource.
 *
 * @param {string} incidentId
 * @param {(event: object) => void} onEvent - called with each parsed JSON event
 * @param {{ onError?: (err) => void, onOpen?: () => void, afterSeq?: number }} [opts]
 * @returns {() => void} unsubscribe function that closes the stream
 * `afterSeq` resumes past already-seen events. The browser only sends the
 * `Last-Event-ID` header on its own automatic reconnects - a deliberate
 * re-subscribe (navigating back to the console mid-run) cannot set headers on an
 * EventSource, so the resume point travels as a query parameter instead.
 */
export function subscribeToIncident(incidentId, onEvent, opts = {}) {
  const base = API_BASE_URL || '';
  const resume = opts.afterSeq ? `?after_seq=${opts.afterSeq}` : '';
  const url = `${base}/api/incidents/${incidentId}/stream${resume}`;
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
