// Pure presentation helpers (no React / MUI imports), so they are trivial to
// unit-test in isolation and cheap to import anywhere.

// Repo status -> MUI Chip `color`.
export function repoStatusColor(status) {
  switch (status) {
    case 'pr_open':
      return 'success';
    case 'migrating':
      return 'info';
    case 'reviewing':
      return 'secondary';
    case 'awaiting_approval':
      return 'warning';
    case 'escalated':
      return 'warning';
    case 'rejected':
    case 'failed':
      return 'error';
    case 'queued':
    default:
      return 'default';
  }
}

// Job status -> MUI Chip `color`.
export function jobStatusColor(status) {
  switch (status) {
    case 'done':
      return 'success';
    case 'failed':
      return 'error';
    case 'running':
      return 'info';
    case 'planning':
      return 'warning';
    default:
      return 'default';
  }
}

// Execution mode -> MUI color. Live modes are a "hot" green, stub is neutral grey.
export function modeColor(mode) {
  return mode === 'live-sdk' || mode === 'live-gemini' ? 'success' : 'default';
}

// Activity tone -> MUI palette color name (used for icon / accent).
export function toneColor(tone) {
  switch (tone) {
    case 'success':
      return 'success';
    case 'warning':
      return 'warning';
    case 'error':
      return 'error';
    default:
      return 'info';
  }
}

/**
 * Parse a unified diff string into classified lines so the viewer can color
 * additions green and deletions red without a syntax highlighter.
 *
 * @param {string} diff
 * @returns {{ type: 'add'|'del'|'meta'|'hunk'|'context', text: string }[]}
 */
export function diffLines(diff) {
  if (!diff || typeof diff !== 'string') return [];
  return diff.split('\n').map((line) => {
    if (line.startsWith('+++') || line.startsWith('---') || line.startsWith('diff ')) {
      return { type: 'meta', text: line };
    }
    if (line.startsWith('@@')) {
      return { type: 'hunk', text: line };
    }
    if (line.startsWith('+')) {
      return { type: 'add', text: line };
    }
    if (line.startsWith('-')) {
      return { type: 'del', text: line };
    }
    return { type: 'context', text: line };
  });
}

// Format a USD amount with 4 decimals of precision (agent runs are cheap).
export function formatUsd(n) {
  return `$${Number(n || 0).toFixed(4)}`;
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
