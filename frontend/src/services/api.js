export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

export async function apiRequest(path, options = {}, token = '') {
  const headers = new Headers(options.headers || {});
  if (options.body && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }
  if (token) headers.set('Authorization', `Bearer ${token}`);

  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || payload.error || `Request failed (${response.status})`);
  return payload;
}

export const login = (username, password) =>
  apiRequest('/api/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) });

export const signup = (name, email, password) =>
  apiRequest('/api/auth/signup', { method: 'POST', body: JSON.stringify({ name, email, password }) });

export const requestPasswordReset = (email) =>
  apiRequest('/api/auth/forgot-password', { method: 'POST', body: JSON.stringify({ email }) });

export const resetPassword = (token, password) =>
  apiRequest('/api/auth/reset-password', { method: 'POST', body: JSON.stringify({ token, password }) });

export const saveScan = (scan, token) =>
  apiRequest('/api/scans', { method: 'POST', body: JSON.stringify(scan) }, token);

export const saveReport = (report, token) =>
  apiRequest('/api/reports', { method: 'POST', body: JSON.stringify(report) }, token);

export const emailReport = (report, token) =>
  apiRequest('/api/reports/email', { method: 'POST', body: JSON.stringify(report) }, token);

export const fetchReports = (token, patientId = '') =>
  apiRequest(`/api/reports${patientId ? `?patient_id=${encodeURIComponent(patientId)}` : ''}`, {}, token);

export const fetchScans = (token, patientId = '') =>
  apiRequest(`/api/scans${patientId ? `?patient_id=${encodeURIComponent(patientId)}` : ''}`, {}, token);

export const fetchAuditEvents = token => apiRequest('/api/audit', {}, token);

export const logout = token => apiRequest('/api/auth/logout', { method: 'POST' }, token);
