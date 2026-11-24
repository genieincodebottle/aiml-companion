import axios from 'axios';

// Who is acting. With login on (MC_OPERATORS set on the backend) that is a bearer
// token kept for this browser tab only. With login off it is just a name, sent in
// X-Operator, which nothing verifies.
const TOKEN_KEY = 'mc_token';
const OPERATOR_KEY = 'mc_operator';

function read(storage, key) {
  try {
    return storage.getItem(key) || '';
  } catch {
    return '';
  }
}

function write(storage, key, value) {
  try {
    if (value) storage.setItem(key, value);
    else storage.removeItem(key);
  } catch {
    /* private mode: keep working without persistence */
  }
}

export const getToken = () => read(sessionStorage, TOKEN_KEY);
export const setToken = (t) => write(sessionStorage, TOKEN_KEY, t);
export const getOperatorName = () => read(localStorage, OPERATOR_KEY) || 'ana.analyst';
export const setOperatorName = (n) => write(localStorage, OPERATOR_KEY, n);

const client = axios.create({ baseURL: '/api' });
client.interceptors.request.use((config) => {
  const token = getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  else config.headers['X-Operator'] = getOperatorName();
  return config;
});

// The backend's error text is written for people; show it rather than "Request failed".
export function errorText(err) {
  return err?.response?.data?.detail || err?.message || 'request failed';
}

export const getHealth = () => client.get('/health').then((r) => r.data);
export const getMe = () => client.get('/me').then((r) => r.data);
export const getAlerts = () => client.get('/alerts').then((r) => r.data.alerts);
export const getAgents = () => client.get('/agents').then((r) => r.data.agents);
export const getCases = () => client.get('/investigations').then((r) => r.data.investigations);

export const startInvestigation = (alert_id, inject_commit_fault = false) =>
  client.post('/investigations', { alert_id, inject_commit_fault }).then((r) => r.data);

export const decide = (workflow_id, decision, note) =>
  client.post('/decisions', { workflow_id, decision, note }).then((r) => r.data);

export const getView = (workflow_id) =>
  client.get(`/investigations/${workflow_id}`).then((r) => r.data);

export const getEvents = (workflow_id) =>
  client.get(`/investigations/${workflow_id}/events`).then((r) => r.data.events);

// EventSource cannot send headers; the stream is read-only and needs none.
export const streamUrl = (workflow_id) => `/api/investigations/${workflow_id}/stream`;
