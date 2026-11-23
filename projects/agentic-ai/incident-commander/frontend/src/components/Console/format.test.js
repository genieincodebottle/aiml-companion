import { describe, it, expect } from 'vitest';

import { riskColor, statusColor, formatTime } from './format';

describe('format helpers', () => {
  it('maps action risk to MUI chip colors', () => {
    expect(riskColor('high')).toBe('error');
    expect(riskColor('medium')).toBe('warning');
    expect(riskColor('low')).toBe('success');
    expect(riskColor('unknown')).toBe('default');
  });

  it('maps incident status to MUI colors', () => {
    expect(statusColor('resolved')).toBe('success');
    expect(statusColor('escalated')).toBe('error');
    expect(statusColor('failed')).toBe('error');
    expect(statusColor('awaiting_approval')).toBe('warning');
    expect(statusColor('detected')).toBe('info');
    expect(statusColor('anything-else')).toBe('primary');
  });

  it('formats timestamps and degrades gracefully', () => {
    expect(formatTime(null)).toBe('');
    expect(formatTime(undefined)).toBe('');
    expect(formatTime(1_700_000_000).length).toBeGreaterThan(0);
  });
});
