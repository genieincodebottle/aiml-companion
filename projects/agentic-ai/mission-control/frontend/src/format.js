// Presentation helpers for the console. Kept pure + tested.

export const STATUS_META = {
  pending: { label: 'Pending', kind: 'neutral' },
  running: { label: 'Running', kind: 'info' },
  awaiting_approval: { label: 'Awaiting approval', kind: 'warn' },
  resolved: { label: 'Resolved', kind: 'ok' },
  escalated: { label: 'Escalated', kind: 'warn' },
  compensated: { label: 'Compensated', kind: 'warn' },
  failed: { label: 'Failed', kind: 'bad' },
  dead_lettered: { label: 'Dead-lettered', kind: 'bad' },
};

export function statusMeta(status) {
  return STATUS_META[status] || { label: status || 'unknown', kind: 'neutral' };
}

export function usd(n) {
  if (typeof n !== 'number') return '$0.000000';
  return '$' + n.toFixed(6);
}

// A short, human label for an event row in the timeline.
export function eventLabel(type) {
  return type.replace(/_/g, ' ');
}

// Which plane an event belongs to - drives the colored dot in the timeline.
export function eventPlane(type) {
  if (
    type.startsWith('gateway') ||
    type === 'cache_hit' ||
    type === 'budget_exceeded' ||
    type === 'provider_failover'
  )
    return 'gateway';
  if (type.startsWith('policy') || type === 'injection_blocked') return 'policy';
  if (type.startsWith('approval') || type === 'maker_checker_blocked') return 'human';
  if (type === 'saga_compensated' || type === 'dead_lettered' || type === 'retry')
    return 'reliability';
  return 'runtime';
}

// "approved by rex at 14:02", or why nobody has: the audit line on a case.
export function decidedBy(view) {
  if (!view?.decision) {
    return view?.status === 'awaiting_approval' ? 'waiting for a reviewer' : '-';
  }
  if (view.decided_by === 'policy:auto_approve') return 'approved by policy (auto-approve was on)';
  const verb = view.decision === 'approve' ? 'approved' : 'rejected';
  const at = view.decided_at ? ` at ${new Date(view.decided_at * 1000).toLocaleTimeString()}` : '';
  return `${verb} by ${view.decided_by}${at}`;
}
