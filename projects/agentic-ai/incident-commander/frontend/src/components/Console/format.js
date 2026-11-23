// Pure presentation helpers (no React / MUI imports), so they are trivial to
// unit-test in isolation and cheap to import anywhere.

// Risk chip color mapping for MUI Chip `color` prop.
export function riskColor(risk) {
  switch (risk) {
    case 'high':
      return 'error';
    case 'medium':
      return 'warning';
    case 'low':
      return 'success';
    default:
      return 'default';
  }
}

// Incident status -> MUI color name.
export function statusColor(status) {
  switch (status) {
    case 'resolved':
      return 'success';
    case 'escalated':
    case 'failed':
      return 'error';
    case 'awaiting_approval':
      return 'warning';
    case 'detected':
      return 'info';
    default:
      return 'primary';
  }
}

// Format an ISO ts or epoch (seconds) into a short local time string.
export function formatTime(ts) {
  if (!ts) return '';
  try {
    const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
    if (Number.isNaN(d.getTime())) return String(ts);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return String(ts);
  }
}
