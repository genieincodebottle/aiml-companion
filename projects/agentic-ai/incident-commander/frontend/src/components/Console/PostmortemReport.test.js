import { describe, it, expect } from 'vitest';

import { decidedBy } from './PostmortemReport';

describe('decidedBy (approval audit line)', () => {
  it('names the human who approved or rejected', () => {
    expect(decidedBy({ decided_by: 'alice', status: 'executed' })).toBe('approved by alice');
    expect(decidedBy({ decided_by: 'bob', status: 'rejected' })).toBe('rejected by bob');
  });

  it('distinguishes policy and timeout from a person', () => {
    expect(decidedBy({ decided_by: 'policy:auto_approve' })).toBe('auto-approved by policy');
    expect(decidedBy({ decided_by: 'nobody:timeout' })).toBe('no decision (approval timed out)');
  });

  it('says nothing when no human gate was involved', () => {
    expect(decidedBy({ status: 'rejected' })).toBe('');
  });
});
