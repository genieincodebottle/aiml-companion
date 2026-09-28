import { describe, it, expect } from 'vitest';

import {
  repoStatusColor,
  jobStatusColor,
  modeColor,
  diffLines,
  formatUsd,
  formatTime,
} from './format';

describe('format helpers', () => {
  it('maps repo status to MUI chip colors', () => {
    expect(repoStatusColor('pr_open')).toBe('success');
    expect(repoStatusColor('migrating')).toBe('info');
    expect(repoStatusColor('reviewing')).toBe('secondary');
    expect(repoStatusColor('awaiting_approval')).toBe('warning');
    expect(repoStatusColor('escalated')).toBe('warning');
    expect(repoStatusColor('rejected')).toBe('error');
    expect(repoStatusColor('failed')).toBe('error');
    expect(repoStatusColor('queued')).toBe('default');
    expect(repoStatusColor('anything-else')).toBe('default');
  });

  it('maps job status to MUI chip colors', () => {
    expect(jobStatusColor('done')).toBe('success');
    expect(jobStatusColor('failed')).toBe('error');
    expect(jobStatusColor('running')).toBe('info');
    expect(jobStatusColor('planning')).toBe('warning');
    expect(jobStatusColor(null)).toBe('default');
  });

  it('maps execution mode to a color', () => {
    expect(modeColor('live-sdk')).toBe('success');
    expect(modeColor('stub')).toBe('default');
    expect(modeColor(undefined)).toBe('default');
  });

  it('classifies unified diff lines by leading marker', () => {
    const diff = [
      'diff --git a/app.py b/app.py',
      '--- a/app.py',
      '+++ b/app.py',
      '@@ -1,3 +1,3 @@',
      '-import old',
      '+import new',
      ' unchanged line',
    ].join('\n');

    const parsed = diffLines(diff);
    expect(parsed).toHaveLength(7);
    expect(parsed[0].type).toBe('meta');
    expect(parsed[1].type).toBe('meta');
    expect(parsed[2].type).toBe('meta');
    expect(parsed[3].type).toBe('hunk');
    expect(parsed[4].type).toBe('del');
    expect(parsed[5].type).toBe('add');
    expect(parsed[6].type).toBe('context');
  });

  it('handles empty or non-string diffs', () => {
    expect(diffLines('')).toEqual([]);
    expect(diffLines(null)).toEqual([]);
    expect(diffLines(undefined)).toEqual([]);
  });

  it('formats USD amounts and degrades gracefully', () => {
    expect(formatUsd(0)).toBe('$0.0000');
    expect(formatUsd(1.23456)).toBe('$1.2346');
    expect(formatUsd(null)).toBe('$0.0000');
  });

  it('formats timestamps and degrades gracefully', () => {
    expect(formatTime(null)).toBe('');
    expect(formatTime(undefined)).toBe('');
    expect(formatTime(1_700_000_000).length).toBeGreaterThan(0);
  });
});
