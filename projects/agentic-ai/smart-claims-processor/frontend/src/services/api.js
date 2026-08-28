import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({ baseURL: API_URL });

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(err);
  }
);

// ── Auth ──────────────────────────────────────────────────────────────────────
export const authAPI = {
  login: (username, password) => api.post('/api/auth/login', { username, password }),
  register: (data) => api.post('/api/auth/register', data),
  logout: () => api.post('/api/auth/logout'),
  getCurrentUser: () => api.get('/api/auth/current-user'),
  getAllUsers: () => api.get('/api/auth/users'),
  updateUser: (id, data) => api.put(`/api/auth/users/${id}`, data),
  deleteUser: (id) => api.delete(`/api/auth/users/${id}`),
  updatePassword: (current, newPass) => api.put('/api/auth/password', { current_password: current, new_password: newPass }),
};

// ── Claims ────────────────────────────────────────────────────────────────────
export const claimsAPI = {
  submitClaim: (data) => api.post('/api/claims/submit', data),
  getAllClaims: (status, limit = 100) => api.get('/api/claims/all', { params: { status, limit } }),
  getClaimById: (id) => api.get(`/api/claims/${id}`),
  getUserClaims: (userId) => api.get(`/api/claims/user/${userId}`),
  processClaim: (id) => api.post(`/api/claims/${id}/process`),
  getClaimStatus: (id) => api.get(`/api/claims/${id}/status`),
};

// ── HITL ──────────────────────────────────────────────────────────────────────
export const hitlAPI = {
  getQueue: (status = 'pending') => api.get('/api/hitl/queue', { params: { status } }),
  getTicket: (ticketId) => api.get(`/api/hitl/ticket/${ticketId}`),
  submitDecision: (ticketId, data) => api.post(`/api/hitl/decide/${ticketId}`, data),
  getStats: () => api.get('/api/hitl/stats'),
};

// ── Appeals ───────────────────────────────────────────────────────────────────
export const appealsAPI = {
  submitAppeal: (data) => api.post('/api/appeals/submit', data),
  getPendingAppeals: (limit = 50) => api.get('/api/appeals/pending', { params: { limit } }),
  getAllAppeals: (status, limit = 100) => api.get('/api/appeals/all', { params: { status, limit } }),
  getUserAppeals: (userId) => api.get(`/api/appeals/user/${userId}`),
  reviewAppeal: (id, data) => api.post(`/api/appeals/${id}/review`, data),
  getAppealById: (id) => api.get(`/api/appeals/${id}`),
};

// ── Policies ─────────────────────────────────────────────────────────────
export const policyAPI = {
  list: () => api.get('/api/policies/'),
  defaults: () => api.get('/api/policies/defaults'),
  upsert: (data) => api.post('/api/policies/', data),
  delete: (policyNumber) => api.delete(`/api/policies/${encodeURIComponent(policyNumber)}`),
};

// ── Analytics ─────────────────────────────────────────────────────────────────
export const analyticsAPI = {
  getMetrics: () => api.get('/api/analytics/metrics'),
  getPipelineStats: () => api.get('/api/analytics/pipeline'),
  getCostBreakdown: (days = 30) => api.get('/api/analytics/costs', { params: { days } }),
  getFraudTrends: (days = 30) => api.get('/api/analytics/fraud-trends', { params: { days } }),
  getHITLMetrics: () => api.get('/api/analytics/hitl'),
  getEvaluationScores: (limit = 50) => api.get('/api/analytics/evaluations', { params: { limit } }),
};

export default api;
